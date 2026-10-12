// Candidate-only SOFA-independent fast coarse Beam/SDF kernel.
// Raw data follows production LiveBeamSDFAdapter / SignedDistanceGrid layout.
// This C ABI is intentionally versioned; Python performs strict parity checks.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
struct V {
    double x, y, z;
    V operator+(V b) const { return {x+b.x,y+b.y,z+b.z}; }
    V operator-(V b) const { return {x-b.x,y-b.y,z-b.z}; }
    V operator*(double t) const { return {x*t,y*t,z*t}; }
};
double norm(V a) { return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z); }
double dot(V a,V b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
V cross(V a,V b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
bool finite(V v) { return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z); }
V quat_rotate(V v, const double* quat, bool inverse=false) {
    double x=quat[0], y=quat[1], z=quat[2], w=quat[3];
    double n=std::sqrt(x*x+y*y+z*z+w*w);
    if (!(n>1e-15) || !std::isfinite(n)) return {NAN,NAN,NAN};
    x/=n; y/=n; z/=n; w/=n;
    if (inverse) {x=-x;y=-y;z=-z;}
    V q{x,y,z};
    V t=cross(q,v)*2.0;
    return v+t*w+cross(q,t);
}
double interpolate(const float* field, const int32_t* dims,
                   const double* origin, const double* spacing, V point) {
    const int nx=dims[0],ny=dims[1],nz=dims[2];
    const double p[3]={point.x,point.y,point.z};
    double frac[3];
    int lo[3];
    for(int axis=0;axis<3;++axis) {
        double v=(p[axis]-origin[axis])/spacing[axis];
        if (!std::isfinite(v)||v<0||v>dims[axis]-1) return INFINITY;
        lo[axis]=std::clamp(static_cast<int>(std::floor(v)),0,dims[axis]-2);
        frac[axis]=std::clamp(v-lo[axis],0.0,1.0);
    }
    int x=lo[0],y=lo[1],z=lo[2];
    auto at=[&](int dx,int dy,int dz)->double{
        return static_cast<double>(field[(static_cast<size_t>(z+dz)*ny+(y+dy))*nx+(x+dx)]);
    };
    double fx=frac[0],fy=frac[1],fz=frac[2];
    double c00=at(0,0,0)*(1-fx)+at(1,0,0)*fx;
    double c10=at(0,1,0)*(1-fx)+at(1,1,0)*fx;
    double c01=at(0,0,1)*(1-fx)+at(1,0,1)*fx;
    double c11=at(0,1,1)*(1-fx)+at(1,1,1)*fx;
    double c0=c00*(1-fy)+c10*fy;
    double c1=c01*(1-fy)+c11*fy;
    return c0*(1-fz)+c1*fz;
}
} // namespace

// out: minimum_clearance_m, lower_bound_m, worst_index,
// scanned_samples, invalid_grid_flag, max_curve_gap_m, safe_flag
// return 0 success, negative return indicates invalid arguments/state.
extern "C" int mcr_fast_coarse_sdf_v1(
    const double* state, int32_t n_states,
    const int32_t* nodes, const double* rest_lengths, int32_t n_elements,
    const float* sdf, const int32_t* dimensions_xyz,
    const double* grid_origin, const double* grid_spacing,
    const double* asset_transform_xyzw, const double* asset_offset,
    double source_to_sim_scale, double catheter_radius,
    double sample_spacing, double switch_margin,
    double slope_allowance, double error_cushion,
    double* out
) noexcept {
    if (!state||!nodes||!rest_lengths||!sdf||!dimensions_xyz||
        !grid_origin||!grid_spacing||!asset_transform_xyzw||
        !asset_offset||!out||n_states<2||n_elements<1||
        dimensions_xyz[0]<2||dimensions_xyz[1]<2||dimensions_xyz[2]<2||
        !(source_to_sim_scale>0)||!(sample_spacing>0)||
        !(slope_allowance>0)||!(error_cushion>=0)||!(catheter_radius>=0)||
        !(switch_margin>0)) return -1;
    for (int i=0;i<3;++i)
        if (!(grid_spacing[i]>0)||!std::isfinite(grid_spacing[i])) return -2;
    double minimum=INFINITY, max_gap=0.0;
    int64_t sampled=0, worst=-1;
    bool invalid=false, definitely_unsafe=false;
    V translation{asset_transform_xyzw[0]+asset_offset[0],
                  asset_transform_xyzw[1]+asset_offset[1],
                  asset_transform_xyzw[2]+asset_offset[2]};
    const double* rotation=asset_transform_xyzw+3;
    auto sample=[&](V sim)->void {
        V source=quat_rotate(sim-translation,rotation,true)*(1.0/source_to_sim_scale);
        double signed_source=interpolate(sdf,dimensions_xyz,grid_origin,grid_spacing,source);
        if (!std::isfinite(signed_source)) {invalid=true;return;}
        double clearance=-signed_source*source_to_sim_scale-catheter_radius;
        if (!std::isfinite(clearance)){invalid=true;return;}
        if (clearance<minimum){minimum=clearance;worst=sampled;}
        // A point below the switch margin alone disproves safe clearance,
        // regardless of any unobserved interpolated-surface uncertainty.
        if(clearance<switch_margin) definitely_unsafe=true;
    };
    for(int32_t i=0;i<n_elements && !invalid && !definitely_unsafe;++i) {
        int32_t n0=nodes[2*i],n1=nodes[2*i+1];
        double length=rest_lengths[i];
        if(n0<0||n0>=n_states||n1<0||n1>=n_states||
           !(length>0)||!std::isfinite(length)) return -3;
        const double* a=state+7*n0;
        const double* b=state+7*n1;
        V p0{a[0],a[1],a[2]},p3{b[0],b[1],b[2]};
        if(!finite(p0)||!finite(p3))return -4;
        V p1=p0+quat_rotate({length/3.0,0,0},a+3);
        V p2=p3+quat_rotate({-length/3.0,0,0},b+3);
        if(!finite(p1)||!finite(p2))return -5;
        double chord=norm(p3-p0);
        bool linear=dot(p1-p0,p2-p1)<0 && chord<0.4*length;
        double deriv=linear?chord:3.0*std::max({norm(p1-p0),norm(p2-p1),norm(p3-p2)});
        double count=std::ceil(std::max(length,deriv)/sample_spacing);
        if(!std::isfinite(count)||count>10000000.0)return -6;
        int n=std::max(1,static_cast<int>(count));
        max_gap=std::max(max_gap,deriv/n);
        // Don't skip the shared endpoint unless this element really shares
        // it: production adapter drops the first sample after element #0.
        for(int j=(i?1:0);j<=n && !invalid && !definitely_unsafe;++j){
            double t=static_cast<double>(j)/n,u=1.0-t;
            V point=linear?(p0*u+p3*t):
                (p0*(u*u*u)+p1*(3*u*u*t)+p2*(3*u*t*t)+p3*(t*t*t));
            sample(point);
            ++sampled;
        }
    }
    const double allowance=slope_allowance*max_gap/2.0+error_cushion;
    const double lower=minimum-allowance;
    const bool safe=!invalid&&!definitely_unsafe&&std::isfinite(lower)&&lower>=switch_margin;
    out[0]=std::isfinite(minimum)?minimum:NAN;
    out[1]=!invalid&&std::isfinite(lower)?lower:NAN;
    out[2]=static_cast<double>(worst);
    out[3]=static_cast<double>(sampled);
    out[4]=invalid?1.0:0.0;
    out[5]=max_gap;
    out[6]=safe?1.0:0.0;
    return 0;
}
