# -*- coding: utf-8 -*-
"""
MAML++ Implementation

@misc{antoniou2019trainmaml,
      title={How to train your MAML}, 
      author={Antreas Antoniou and Harrison Edwards and Amos Storkey},
      year={2019},
      eprint={1810.09502},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1810.09502}, 
}
https://arxiv.org/abs/1810.09502

Adapted from https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
"""
import torch
from torch import nn

from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module


class LayerNorm(nn.Module):
    """Layer normalization for MAML++"""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            x = x * self.weight + self.bias

        return x


class MAMLPlusPlusLayer(nn.Module):
    def __init__(self, feat_dim=64, way_num=5, use_layer_norm=True) -> None:
        super(MAMLPlusPlusLayer, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.linear = nn.Linear(feat_dim, way_num)
        if self.use_layer_norm:
            self.layer_norm = LayerNorm(feat_dim)

    def forward(self, x):
        if self.use_layer_norm:
            x = self.layer_norm(x)

        return self.linear(x)


class MAMLPlusPlus(MetaModel):
    def __init__(self, inner_param, feat_dim, use_layer_norm=True, 
                 use_per_param_lr=True, use_mslo=True, **kwargs):
        super(MAMLPlusPlus, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = MAMLPlusPlusLayer(feat_dim, way_num=self.way_num, 
                                           use_layer_norm=use_layer_norm)
        self.inner_param = inner_param
        
        self.use_layer_norm = use_layer_norm
        self.use_per_param_lr = use_per_param_lr
        self.use_mslo = use_mslo
        
        self.enable_inner_loop_optimizable_bn_params = inner_param.get("enable_inner_loop_optimizable_bn_params", False)
        self.learnable_bn_gamma = inner_param.get("learnable_bn_gamma", True)
        self.learnable_bn_beta = inner_param.get("learnable_bn_beta", True)
        self.first_order_to_second_order_epoch = inner_param.get("first_order_to_second_order_epoch", -1)
        self.second_order = inner_param.get("second_order", True)
        self.multi_step_loss_num_epochs = inner_param.get("multi_step_loss_num_epochs", 15)
        
        self._current_epoch = 0
        
        if self.use_per_param_lr:
            self.init_per_param_lr()
            
        if self.use_mslo:
            self.mslo_lambda = inner_param.get("mslo_lambda", 0.1)
        
        convert_maml_module(self)

    def init_per_param_lr(self):
        self.per_param_lr = nn.ParameterDict()
        
        param_names = []
        for name, param in self.named_parameters():
            param_names.append(name)
        
        for name in param_names:
            base_lr = self.inner_param.get("lr", 0.01)
            lr_param = nn.Parameter(torch.tensor(base_lr, dtype=torch.float32))
            param_name = name.replace('.', '_')
            self.per_param_lr[param_name] = lr_param

    def get_learning_rate(self, param_name):
        if self.use_per_param_lr:
            param_name = param_name.replace('.', '_')
            if param_name in self.per_param_lr:
                return torch.abs(self.per_param_lr[param_name])  
            else:
                return self.inner_param.get("lr", 0.01)
        else:
            return self.inner_param.get("lr", 0.01)

    def forward_output(self, x):
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)

        return out2

    def set_forward(self, batch):
        image, global_target = batch  
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))

        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch  
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        mslo_losses = []
        
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            
            if self.use_mslo:
                inner_losses = self.set_forward_adaptation_with_mslo(
                    episode_support_image, episode_support_target)
                mslo_losses.extend(inner_losses)
            else:
                self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        
        main_loss = self.loss_func(output, query_target.contiguous().view(-1))
        
        if self.use_mslo and mslo_losses:
            mslo_loss = torch.mean(torch.stack(mslo_losses))
            total_loss = main_loss + self.mslo_lambda * mslo_loss
        else:
            total_loss = main_loss
            
        acc = accuracy(output, query_target.contiguous().view(-1))
        
        return output, acc, total_loss

    def set_forward_adaptation(self, support_set, support_target):
        model_parameters = []
        param_names = []
        for name, param in self.named_parameters():
            if not name.startswith('per_param_lr'):
                if 'bn' in name or 'batch_norm' in name:
                    if self.enable_inner_loop_optimizable_bn_params:
                        if ('weight' in name and self.learnable_bn_gamma) or \
                           ('bias' in name and self.learnable_bn_beta):
                            model_parameters.append(param)
                            param_names.append(name)
                    continue
                else:
                    model_parameters.append(param)
                    param_names.append(name)
        
        for parameter in model_parameters:
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()
        
        num_steps = (self.inner_param["train_iter"] if self.training 
                    else self.inner_param["test_iter"])
        
        for i in range(num_steps):
            output = self.forward_output(support_set)
            loss = self.loss_func(output, support_target)
            
            create_graph = self.second_order and not self.is_first_order_epoch(getattr(self, '_current_epoch', 0))
            
            grad = torch.autograd.grad(loss, model_parameters, create_graph=create_graph, allow_unused=True)

            for k, (param, name) in enumerate(zip(model_parameters, param_names)):
                if grad[k] is not None:
                    lr = self.get_learning_rate(name)
                    if param.fast is None:
                        param.fast = param - lr * grad[k]
                    else:
                        param.fast = param.fast - lr * grad[k]

    def set_forward_adaptation_with_mslo(self, support_set, support_target):
        """MAML++ adaptation with Multi-Step Loss Optimization"""
        model_parameters = []
        param_names = []
        for name, param in self.named_parameters():
            if not name.startswith('per_param_lr'):
                if 'bn' in name or 'batch_norm' in name:
                    if self.enable_inner_loop_optimizable_bn_params:
                        if ('weight' in name and self.learnable_bn_gamma) or \
                           ('bias' in name and self.learnable_bn_beta):
                            model_parameters.append(param)
                            param_names.append(name)
                    continue
                else:
                    model_parameters.append(param)
                    param_names.append(name)
        
        for parameter in model_parameters:
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()
        
        num_steps = (self.inner_param["train_iter"] if self.training 
                    else self.inner_param["test_iter"])
        
        inner_losses = []
        
        for i in range(num_steps):
            output = self.forward_output(support_set)
            loss = self.loss_func(output, support_target)
            inner_losses.append(loss)
            
            create_graph = self.second_order and not self.is_first_order_epoch(getattr(self, '_current_epoch', 0))
            
            grad = torch.autograd.grad(loss, model_parameters, create_graph=create_graph, allow_unused=True)

            for k, (param, name) in enumerate(zip(model_parameters, param_names)):
                if grad[k] is not None:
                    lr = self.get_learning_rate(name)
                    if param.fast is None:
                        param.fast = param - lr * grad[k]
                    else:
                        param.fast = param.fast - lr * grad[k]
                    
        return inner_losses
    
    def is_first_order_epoch(self, epoch):
        if self.first_order_to_second_order_epoch > 0:
            return epoch < self.first_order_to_second_order_epoch
        return not self.second_order

    def set_current_epoch(self, epoch):
        self._current_epoch = epoch
        
        if hasattr(self, 'multi_step_loss_num_epochs'):
            if epoch < self.multi_step_loss_num_epochs:
                self.use_mslo = True
            else:
                self.use_mslo = self.inner_param.get("use_mslo", True)
        
        if self.first_order_to_second_order_epoch > 0 and epoch >= self.first_order_to_second_order_epoch:
            self.second_order = True