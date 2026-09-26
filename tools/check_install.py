"""Runtime-only installed-model and generated-video smoke test."""
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import numpy as np
import av

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.registry import model_specs
from app.runtime import Detector
from app.video import process_video
from app.app import create_app


def main():
    specs=model_specs()
    for spec in specs:
        with Detector(spec) as detector: detector.detect(np.zeros((480,640,3),dtype=np.uint8))
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=directory/'generated.mp4'
        with av.open(str(source),'w') as writer:
            stream=writer.add_stream('libx264',rate=25)
            stream.width,stream.height,stream.pix_fmt=64,48,'yuv420p'
            for i in range(3):
                frame=av.VideoFrame.from_ndarray(np.full((48,64,3),80+i,dtype=np.uint8),format='bgr24')
                frame.pts,frame.time_base=i,Fraction(1,25)
                for packet in stream.encode(frame): writer.mux(packet)
            for packet in stream.encode(): writer.mux(packet)
        original,result,_=process_video(source,directory/'result.mp4',ROOT/'models')
        assert result.frames==3 and result.timeline_hash==original.timeline_hash
        app=create_app(runtime=directory/'jobs')
        assert app.test_client().get('/').status_code==200
        app.extensions['video_jobs'].close()
    assert not any(name in sys.modules for name in ('torch','mmcv','mmdet','tensorrt'))
    print('PASS: all registered ONNX models, CPU inference, generated full video, frontend; no training/TensorRT imports or dataset/checkpoint access.')


if __name__=='__main__': main()
