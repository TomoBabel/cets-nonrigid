"""Internal fit-loader adapter for CETS bundles.

The exchange API is :mod:`cets_nonrigid.api`. Native numerical IR objects are
transient views; they cannot be serialized as standalone deformation stores.
"""
from types import SimpleNamespace

class DeformationStore:
    @staticmethod
    def read(path):
        from cets_nonrigid.api import read_bundle
        from cets_nonrigid.runtime import runtime_ir
        bundle = read_bundle(path)
        return SimpleNamespace(ir=runtime_ir(bundle), native_files={})

    @staticmethod
    def write(path, bundle, **kwargs):
        from cets_nonrigid.api import AlignmentBundle, write_bundle
        if not isinstance(bundle, AlignmentBundle) or kwargs:
            raise ValueError("standalone numerical IR stores are unsupported; use to_cets and write_bundle")
        return write_bundle(bundle, path)
