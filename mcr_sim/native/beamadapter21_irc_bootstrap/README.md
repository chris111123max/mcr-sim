# Production SOFA21 IRC initialization compatibility library

This is the exact previously validated isolated BeamAdapter build, promoted without changing the global SOFA installation. manifest.json records the binary SHA256. sofa_compat.py requires this library and rejects any simultaneous mapping of the old library.

IRC_bootstrap.patch handles the initial single-old-node case by copying that sole pose instead of indexing a nonexistent old segment. It does not alter physics integration, contact, precision thresholds, insertion limits or position during physical stepping. The patched complete controller header, original build commands, original installation hashes and regression report are retained here for provenance.

The separate production singleton ConstantForceFields preserve weighted nodal magnetic torques; they do not multiply the full 4mm magnetic moment.
