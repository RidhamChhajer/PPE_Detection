# Adding a detector

For earplug-v1, evaluate a compatible RTMDet model and add one registry entry:
`id`, `display_name`, `version`, `enabled`, `dataset_version`, ordered `labels`,
BGR `colors`, `input_size`, confidence/NMS thresholds, `backends`, `contract`,
ONNX filename/hash and checkpoint filename/hash. Local TensorRT metadata is optional.

Export with embedded model ID/labels, verify parity, install and enable the entry.
Frontend/video processing discovers it automatically. No class-name branches are
needed. The current contract is 640x640 RTMDet; another network/output contract
needs an explicit adapter and tests rather than guessed output interpretation.
COCO category IDs are separate from zero-based network class IDs.
Run registry, inference and full-video smoke tests after adding a model.
