# Limitations

Training diversity, annotation quality, camera view, lighting and object size
determine performance. Versioning does not improve accuracy. The goggles test
has only five images; footwear has many false positives and low
NOT_SAFETY_SHOE recall. See [measured results](MODELS.md).

Collect independent workers, cameras/sessions and reviewed hard negatives. Do not
tune on held-out tests or treat a familiar factory clip as independent validation.
A visual shoe category does not certify its protective standard. The application
makes no industrial, compliance or safety-certification accuracy claim.

Videos are processed locally one job at a time, up to 512 MiB, with one video
stream, increasing timestamps, even dimensions >=16 pixels, <=8.3 megapixels and
<=240 FPS. Normalize rotation metadata first. Output is silent. Uploads are
removed after processing; app outputs expire after 24 hours and job metadata
does not survive restarts. Download results before restarting.

No worker identity/tracking is used. Review annotated results manually. CPU can
process slower than playback; other OS/GPU environments need local verification.
Rebuild TensorRT on the final hardware. ONNX CPU is the portable default.
