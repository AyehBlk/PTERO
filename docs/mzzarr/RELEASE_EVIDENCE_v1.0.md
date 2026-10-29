# mzZarr v1.0 ? Release Evidence and Limitations

**Evidence snapshot:** 29 October 2026
**Scope:** PTERO reference implementation and mzZarr v1.0 format

This document records verification evidence. It is not normative and
does not replace `SPECIFICATION.md`.

## Verified implementation baseline

Step 5E.8G Part 15B completed successfully on Windows with:

- Python 3.13.2
- pytest 9.0.3
- Zarr 3.2.1
- 515 collected and passed tests
- 526 warnings
- Full-suite duration: 60.24 seconds

The dedicated streaming-writer tests passed 3/3.
The targeted mzZarr regression selection passed 70/70.

The generic streaming writer now rejects a nonempty 4D batch lacking
per-peak ion mobility before growing its peak arrays. Tests also
confirm that empty batches without mobility remain allowed and that
supplied mobility values are preserved.

Verified SHA-256 for the patched reference-implementation file
`ptero/spectra/backends/zarr_backend.py`:

`9af5c299a900001445c2d520697065d9c829c65ea42e02873b4d35ac5c9878dd`

## Conversion behavior

Synthetic mzML conversion tests demonstrated:

- Rejection of mixed nonempty spectra when per-peak mobility was
  missing, before the destination dataset was created.
- Acceptance of an empty spectrum between nonempty 4D spectra.
- Exact expected flattened peak arrays and CSR offsets in the
  synthetic examples.
- Validation of generated mzZarr v1.0 stores.

These are same-implementation conversion tests, not independent
cross-language interoperability demonstrations.

## Additional scientific evidence

Project verification includes real mzML and Bruker data conversion,
spectrum-count and peak-count checks, source-versus-output comparisons,
and performance measurements.

Some Bruker comparisons use the same vendor decoder on both sides.
They establish internal consistency under that decoder, not
independent verification of the vendor's underlying measurements.

Native scan linkage to downstream identification results establishes
linkage completeness in the tested workflow, not identification
accuracy.

## Interoperability limitations

Synthetic interoperability tests demonstrate standard-library reading
of PTERO-written data, PTERO reading independently constructed data,
mzZarr-only format identifiers, and empty-spectrum support.

The evidence does not yet establish:

- A separately maintained cross-language implementation.
- Independent real-data decoding across all supported vendors.
- Universal support for optional Zarr v3 codecs.
- Formal conformance certification by an external standards body.

A minimal independent 3D/4D conformance example is recommended as a
separate post-freeze publication milestone.

## Release interpretation

A successful test suite verifies the tested implementation state.
It does not automatically prove that every requirement in the
normative specification is enforced by the current validator.

The specification freeze and the software release are separate events.
A public release tag must reference a reviewed, committed repository
state. No release tag is established by this document.
