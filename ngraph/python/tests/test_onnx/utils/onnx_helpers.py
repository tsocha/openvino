# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import numpy as np
import onnx
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from openvino.inference_engine import IECore

import ngraph as ng
from ngraph.impl import Function


def np_dtype_to_tensor_type(data_type: np.dtype) -> int:
    """Return TensorProto type for provided numpy dtype.

    :param data_type: Numpy data type object.
    :return: TensorProto.DataType enum value for corresponding type.
    """
    return NP_TYPE_TO_TENSOR_TYPE[data_type]


def import_onnx_model_bak(model: onnx.ModelProto) -> Function:
    onnx.checker.check_model(model)
    model_byte_string = model.SerializeToString()

    ie = IECore()
    ie_network = ie.read_network(model=model_byte_string, weights=b"", init_from_buffer=True)

    ng_function = ng.function_from_cnn(ie_network)
    return ng_function


def import_onnx_model(model: onnx.ModelProto) -> Function:
    # import pudb
    # pudb.set_trace()
    onnx.checker.check_model(model)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdirname:
        ie = IECore()
        onnx_model_path = Path(tmpdirname) / "model.onnx"
        onnx.save(model, onnx_model_path)
        # print("created temporary directory", tmpdirname)
        # print("saved model:", onnx_model_path)

        import sys
        sys.path.append("../../model-optimizer")
        from mo.main import main
        class tmp_parser:
            batch=None
            data_type='float'
            disable_fusing=None
            disable_gfusing=None
            disable_resnet_optimization=False
            disable_weights_compression=False
            enable_concat_optimization=False
            extensions='extensions'
            finegrain_fusing=None
            framework=None
            freeze_placeholder_with_value=None
            input=None
            input_model=onnx_model_path
            input_shape=None
            log_level='ERROR'
            mean_values=()
            model_name=None
            move_to_preprocess=None
            output=None
            output_dir=tmpdirname
            progress=False
            reverse_input_channels=False
            scale=None
            scale_values=()
            silent=False
            static_shape=False
            transformations_config=None
            def parse_args():
                return tmp_parser

        main(tmp_parser, 'onnx')

        ov_model_path = Path(tmpdirname) / "model.xml"
        ov_weights_path = Path(tmpdirname) / "model.bin"
        ie_network = ie.read_network(model=ov_model_path, weights=ov_weights_path)

        ng_function = ng.function_from_cnn(ie_network)
    return ng_function
