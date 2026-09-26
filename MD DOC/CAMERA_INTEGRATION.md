# Camera integration roadmap

Current input is an uploaded complete video; live capture is not implemented.
`Detector.detect(BGR_uint8_frame)` returns structured detections without camera,
worker or factory-ROI rules.

For two cameras, add capture workers producing camera ID, timestamp and frame
into bounded queues. Keep reconnect/health handling separate from inference.
Use a measured per-GPU scheduler; avoid uncontrolled sharing of TensorRT contexts.
Run every detector on original pixels, then render results. Keep each camera's
timeline/output separate and explicitly decide overload buffering versus dropping.
The upload workflow continues to process every frame.

Camera controls belong to the input/job layer; registration and class rendering
remain unchanged. Benchmark both feeds and validate independent recordings on the
final RTX 3050. Current video checks make no live-camera throughput/accuracy claim.
