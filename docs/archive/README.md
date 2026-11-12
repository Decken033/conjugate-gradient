# Historical archive and cleanup inventory

These files are historical records, not current build or submission instructions. Use the root README and task-specific READMEs for runnable commands. Original private paths, old build targets and source-location references in snapshots are intentionally preserved; do not execute archived recipes unchanged.

- [Handoff run report](task_c_handoff/docs/RUN_REPORT.md) and [run guide](task_c_handoff/docs/RUN_GUIDE.md): original unprofiled WSL experiments and scheduling configurations.
- [Handoff reports and evidence](task_c_handoff/): documents and CSVs restored directly from their original Git blobs at the manifest source revision.
- [Original root README](root_README_3a5afde.md): historical instructions and environment context at 3a5afde. Relative links refer to that revision's root layout.
- `vanda/`: four uncertain historical build/validation/small-scaling recipes and the separately authored 3D reproduction proposal. Their relevance to final claims is unconfirmed. The proposal is not the recovered original 3D submission.
- [Cleanup manifest](cleanup_manifest.json): every removed path, original Git blob object ID and canonical SHA-256 content hash; archive destinations; historical checkout hashes; recorded refs. Removed root MPI safety code differs from restored historical Task B and remains recoverable from 3a5afde and protected integration history.

The historical algorithm document refers to a supplied CG summary document whose original title uses Chinese characters. That title is retained only inside the unchanged historical snapshot; the referenced document was not supplied. It is not required for reproduction.

No archived CSV is an input to new benchmarks. Profiled V0 source provenance and the final Task C report remain unavailable. Do not reinterpret the unprofiled handoff as V0.

## Byte preservation and hash meanings

All 18 snapshots were restored from the raw Git objects at `3a5afde14b41ce393807641dc8ff6c4a6129ad34`, using binary subprocess capture and binary file writes. No newline, encoding or text normalization was applied; the two DOCX files retain their original binary bytes. All 18 files passed direct byte comparison against those Git blobs, and their Git clean-filter hashes matched the original blob object IDs.

The manifest records `source_revision` and the original source path for each snapshot/deletion. `git_blob` is the original Git object ID (SHA-1 with Git object framing in this repository). `sha256` is SHA-256 of the raw canonical file content, without Git framing. These are different hash algorithms over different inputs and are not interchangeable.

`original_checkout_sha256` preserves the earlier Windows checkout digest explicitly: Git checkout newline conversion can make that content differ from its canonical blob. The historical `standalone_sha256` and `existing_evidence_sha256` maps also describe pre-cleanup checkout bytes, as specified by `hash_semantics`; they are not assertions that all current documentation matches that earlier snapshot. No experimental values were changed.

Archive-specific `-text` attributes protect the canonical bytes from newline conversion when added. Verification used `git hash-object --path` without `-w` to apply Git's actual clean transformation without staging or writing Git objects. The real index was unchanged. The 110 deletion records were checked against the exact tracked deletion set and canonical source hashes.
