"""Small generated artifacts for offline runtime tests."""
import json
from pathlib import Path
from app.registry import CONTRACT, sha256


def registry_fixture(root, names=('goggles-v1',), labels=None, real=False):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    models = []
    for name in names:
        classes = labels or [name.split('-')[0]]
        path = root/f'{name}.onnx'
        if real:
            import numpy as np
            import onnx
            from onnx import helper,TensorProto,numpy_helper
            raw = np.zeros((1,8400,4+len(classes)),dtype=np.float32)
            raw[0,0,:4] = [10,10,30,30]
            raw[0,0,4:] = .8
            graph = helper.make_graph([helper.make_node('Constant',[],['detections'],value=numpy_helper.from_array(raw))], 'test',
                [helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,640,640])],
                [helper.make_tensor_value_info('detections',TensorProto.FLOAT,[1,8400,4+len(classes)])])
            model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)])
            model.ir_version = 9
            helper.set_model_props(model,dict(contract=CONTRACT,model_id=name,labels=json.dumps(classes)))
            onnx.save(model,str(path))
        else: path.write_bytes(b'test')
        models.append(dict(id=name,display_name=name,version=name.rsplit('-',1)[1],enabled=True,labels=classes,
            colors=[[0,255,0] for _ in classes],input_size=[640,640],confidence_threshold=.3,nms_iou_threshold=.65,
            dataset_version='test dataset v1',backends=['onnx','tensorrt'],contract=CONTRACT,
            onnx=dict(filename=path.name,sha256=sha256(path)),checkpoint=dict(filename=name+'.pth',sha256='0'*64),tensorrt=None))
    manifest = root/'registry.json'
    manifest.write_text(json.dumps(dict(schema_version=1,release_base_url='',models=models)),encoding='utf-8')
    return manifest
