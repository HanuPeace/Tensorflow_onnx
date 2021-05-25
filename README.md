!pip install  tensorflow==2.3 
!pip install  onnx==1.8
!pip install  onnx-tf==1.8
!pip install tf2onnx ==1.8

!python -m tf2onnx.convert --opset 13 --tflite detect.tflite --output model2.onnx

!pip install --upgrade onnx2keras
import onnx2keras
from onnx2keras import onnx_to_keras
import keras
import onnx

onnx_model = onnx.load('model2.onnx')
k_model = onnx_to_keras(onnx_model, ['data'])

keras.models.save_model(k_model,'kerasModel.h5',overwrite=True,include_optimizer=True)

!pip3 install --upgrade tensorflow
!tflite_convert \  --keras_model_file=kerasModel.h5 \  --output_file=mobilenet.tflite
