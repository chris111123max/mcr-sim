"""Hash-pinned dataset eligibility; only five audited diameter exceptions."""
import json,hashlib,math,struct
from pathlib import Path
APPROVED_DIAMETER_SEEDS=frozenset((101,202,303,505,808))

def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_dataset_asset(directory,metadata):
    directory=Path(directory).resolve()
    if directory.parent.name not in ("train","test"):
        if not metadata.get("validation",{}).get("passed",False):raise ValueError("Vessel metadata is not validated: "+str(directory))
        return
    root=directory.parent.parent
    manifest=json.loads((root/"dataset_manifest.json").read_text())
    split=directory.parent.name;seed=int(metadata["seed"])
    matches=[a for a in manifest[split] if a["seed"]==seed and (root/a["path"]).resolve()==directory]
    if len(matches)!=1:raise ValueError("Dataset split/path/seed mismatch")
    entry=matches[0]
    for name,expected in entry["files_sha256"].items():
        path=(directory/name).resolve()
        if directory not in path.parents or not path.is_file() or _sha(path)!=expected:raise ValueError("Dataset hash/path mismatch: "+str(path))
    if metadata.get("scale_diameter_array")!=[.85] or metadata["meshing"]["voxel_spacing_mm"]!=[.6,.6,.9]:raise ValueError("Wrong dataset scale/resolution")
    q=metadata["validation"];m=q["mesh_quality"]
    if not (m["watertight"] and m["winding_consistent"] and m["connected_components"]==1 and all(m[k]==0 for k in ("boundary_or_nonmanifold_edges","degenerate_triangles","zero_area_triangles","near_degenerate_triangles"))):raise ValueError("Invalid mesh integrity")
    if not (q["all_radii_positive"] and q["centerlines_within_mesh_bounds"] and all(x["connected"] for x in q["branch_connections"].values()) and all(q["vtk_reload"].values()) and all(q["png_reload"].values()) and q["centerlines_inside_mesh"]["status"]=="PASS"):raise ValueError("Invalid non-diameter geometry checks")
    with (directory/"collision_inner.stl").open("rb") as stream:
        stream.seek(80);faces=struct.unpack("<I",stream.read(4))[0]
    if faces!=entry["faces"] or faces!=m["triangles"] or not 3000<=faces<=3300 or (directory/"collision_inner.stl").stat().st_size!=84+50*faces:raise ValueError("Not the audited coarse STL")
    failures=[];sections=0
    for region,items in q["multi_section_diameters"].items():
        for item in items:
            sections+=1;actual=item["diameter_median_mm"]
            if actual is None or not math.isfinite(actual) or not item["coverage_complete"]:raise ValueError("Invalid diameter measurement")
            if not item["within_configured_tolerance"]:
                failures.append({"region":region,"index":item["centerline_index"],"actual_mm":actual,"allowed_mm":[item["allowed_min_mm"],item["allowed_max_mm"]],"excess_mm":max(item["allowed_min_mm"]-actual,actual-item["allowed_max_mm"],0.)})
    if sections!=24:raise ValueError("Expected 24 audited diameter sections")
    if q["passed"]:
        if failures or q["failure_reasons"]:raise ValueError("Inconsistent validation PASS")
    else:
        if not (split=="train" and seed in APPROVED_DIAMETER_SEEDS and entry["dataset_acceptance"]=="USER_ACCEPTED_DIAMETER_DEVIATION" and failures==entry["diameter_deviations"] and failures and len(q["failure_reasons"])==len(failures)):raise ValueError("No exact audited diameter-only exception")
        if any(not reason.startswith(tuple(x["region"]+"[" for x in failures)) for reason in q["failure_reasons"]):raise ValueError("Additional non-diameter failure")
