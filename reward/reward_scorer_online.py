# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : reward_scorer_online.py
#   author   : ***
#   date     : 2025-07-25
#   desc     : 
# ======================================================== 
import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

class RewardScorerOnline:
    """
    通过 gRPC 调用远程 TensorFlow Serving 模型的 Reward Scorer
    """

    def __init__(
        self,
        service_endpoint: str,
        model_name: str,
        signature_name: str = "serving_default",
        input_name: str = "inputs",
        output_name: str = "scores_list",
        timeout: float = 30.0,
        max_batch_size: int = 20,
    ):
        self.service_endpoint = service_endpoint
        self.model_name = model_name
        self.signature_name = signature_name
        self.input_name = input_name
        self.output_name = output_name
        self.timeout = timeout
        self.max_batch_size = max_batch_size

        self._channel = None
        self._stub = None

    @classmethod
    def from_args(cls, args: "RewardArguments"):
        """
        从 RewardArguments 创建 scorer
        """
        return cls(
            service_endpoint=args.reward_service_endpoint,
            model_name=args.reward_service_model_name,
            signature_name=args.reward_service_signature,
            input_name=args.reward_service_input_name,
            output_name=args.reward_service_output_name,
            timeout=args.reward_service_timeout,
            max_batch_size=args.reward_max_batch_size,
        )

    def _get_stub(self):
        """懒加载 gRPC stub"""
        if self._stub is None:
            self._channel = grpc.insecure_channel(self.service_endpoint)
            self._stub = prediction_service_pb2_grpc.PredictionServiceStub(self._channel)
        return self._stub

    def predict(self, reward_input):
        """
        Args:
            reward_input: tf.Tensor 或 np.ndarray, shape=(batch_size, 49), dtype=string

        Returns:
            numpy array, shape=(batch_size, n, 9)
        """
        # 转为 numpy
        if isinstance(reward_input, tf.Tensor):
            reward_input = reward_input.numpy()

        batch_size = reward_input.shape[0]
        all_scores = []

        stub = self._get_stub()

        try:
            for i in range(0, batch_size, self.max_batch_size):
                batch = reward_input[i:i + self.max_batch_size,:]

                request = predict_pb2.PredictRequest()
                request.model_spec.name = self.model_name
                request.model_spec.signature_name = self.signature_name

                request.inputs[self.input_name].CopyFrom(
                    tf.make_tensor_proto(batch, shape=batch.shape, dtype=tf.string)
                )

                response = stub.Predict(request, timeout=self.timeout)
                output_tensor = response.outputs[self.output_name]
                scores = tf.make_ndarray(output_tensor)
                all_scores.append(scores[:,0,:])
            return np.concatenate(all_scores, axis=0)

        except grpc.RpcError as e:
            print(f"[gRPC Error] Code: {e.code()}, Details: {e.details()}")
            raise
        except KeyError:
            print(f"Output tensor '{self.output_name}' not found.")
            print("Available outputs:", list(response.outputs.keys()))
            raise

    def close(self):
        """关闭 gRPC channel"""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    import time
    from transformers import HfArgumentParser
    from parser.tokenizer import UrlTokenizer
    from parser.reward_parser import RewardParser
    from dataset.pt_dataset import PtDataset
    from torch.utils.data import DataLoader
    from arguments.data_args import DataArguments
    from arguments.model_args import ModelArguments
    from arguments.train_args import TrainingArguments
    from arguments.reward_args import RewardArguments

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, RewardArguments))
    model_args, data_args, train_args, reward_args = parser.parse_args_into_dataclasses()
    data_args.file_shuffle = False 
    data_args.inter_file_shuffle = False
    data_args.reward_stage = True
    data_args.reward_batch_size = 100

    # 初始化组件
    url_tokenizer = UrlTokenizer.from_args(data_args)
    reward_parser = RewardParser(data_args.columns_def, reward_args.reward_serving_column_idx, tokenizer=url_tokenizer)
    dataset = PtDataset.from_args(data_args, parser_func=reward_parser.parser, test_root_dirs=data_args.test_root_dirs)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=data_args.num_workers, pin_memory=True)

    # 创建在线 scorer
    scorer = RewardScorerOnline.from_args(reward_args)

    for epoch in range(reward_args.reward_train_epochs):
        for x in dataloader:
            start_time = time.time()
            pt_input, reward_input = x
            print("Input shape:", reward_input.shape)
            print("Input dtype:", reward_input.dtype)
            scores = scorer.predict(reward_input)
            latency = time.time() - start_time
            print(f"Scores shape: {scores.shape}, Latency: {latency * 1000:.2f}ms")
            break
        break

    scorer.close()  # 或使用 with 语句
