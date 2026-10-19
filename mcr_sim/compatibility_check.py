"""Bounded SOFA21 component initialization check; no vessel and no animate()."""
import json
from pathlib import Path
from .config import DEFAULT_CONFIG
from .sofa_compat import load_plugins, message_collector


def main():
    import Sofa
    from .mcr_simulator import Simulator
    report = {"scope": "minimal root pipeline init only; no catheter/vessel integration", "physics_substeps_executed": 0}
    messages = message_collector()
    root = Sofa.Core.Node("compatibility_check")
    try:
        with messages:
            report["actual_sources"] = load_plugins(root)
            sim = Simulator(root)
            report["before_init"] = {"dt_s": root.dt.value, "lcp_class": sim.lcp_solver.getClassName(),
                                     **{name: getattr(sim.lcp_solver, name).value for name in ("mu", "tolerance", "maxIt", "build_lcp")}}
            messages.require_no_errors("component creation")
            Sofa.Simulation.init(root)
            messages.require_no_errors("minimal pipeline init")
            values = {name: getattr(sim.lcp_solver, name).value for name in ("mu", "tolerance", "maxIt", "build_lcp")}
            assert abs(float(values["mu"]) - .1) < 1e-12
            assert abs(float(values["tolerance"]) - 1e-4) < 1e-12
            assert int(values["maxIt"]) == 2000 and not bool(values["build_lcp"])
            assert abs(float(root.dt.value) - .006) < 1e-12
            report["after_init"] = {"dt_s": root.dt.value, **values}
            # Real SOFA object/Data construction, deliberately no full beam init.
            from .mcr_instrument import Instrument
            from .mcr_mag_controller import MagController
            probe = Sofa.Core.Node("uninitialized_beam_component_probe")
            instrument = Instrument(probe, [0, 0, 0, 0, 0, 0, 1])
            controller = MagController(instrument, DEFAULT_CONFIG.instrument.initial_field_t)
            instrument.InstrumentCombined.addObject(controller)
            def read(obj, key):
                value = getattr(obj, key).value
                return value.tolist() if hasattr(value, "tolist") else value
            report["beam_component_data_before_full_init"] = {
                **{key: read(instrument.rest_shape, key) for key in ("straightLength", "length", "densityOfBeams", "numEdgesCollis", "numEdges", "youngModulus", "youngModulusExtremity", "radius", "radiusExtremity")},
                "mechanical_grid_resolution": read(instrument.RG, "n"),
                "magnetic_indices": read(instrument.CFF, "indices"),
                "magnetic_forces": read(instrument.CFF, "forces"),
                "indexFromEnd": read(instrument.CFF, "indexFromEnd"),
                "irc_listening": read(instrument.IRC, "listening"),
                "irc_speed": read(instrument.IRC, "speed"),
                "catheter_models": [instrument.line_model.getClassName(), instrument.point_model.getClassName()],
                "constraint_correction": instrument.correction.getClassName(),
                "rayleighStiffness": read(instrument.ode, "rayleighStiffness"),
                "rayleighMass": read(instrument.ode, "rayleighMass"),
            }
            messages.require_no_errors("uninitialized beam component creation")
            report["beam_full_init"] = "NOT_RUN"
            report["magnetic_torque_execution"] = "NOT_RUN"
            report["status"] = "PASS"
    except Exception:
        import traceback
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["messages"] = messages.messages
        target = Path(__file__).resolve().parent / "reports/sofa21_components.json"
        target.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps(report, indent=2, default=str))
        Sofa.Simulation.unload(root)


if __name__ == "__main__":
    main()
