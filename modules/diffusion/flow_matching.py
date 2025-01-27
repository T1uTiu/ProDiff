import torch
import torch.nn as nn
from modules.flow_matching.path.scheduler import CondOTScheduler
from modules.flow_matching.path import AffineProbPath
from modules.flow_matching.solver import ODESolver
from modules.flow_matching.utils.model_wrapper import ModelWrapper

class FlowMatching(nn.Module):
    def __init__(
            self, out_dims, denoise_fn, 
            num_features=1, 
            step_size=0.05,
            spec_min=None, spec_max=None
        ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.out_dims = out_dims
        self.num_features = num_features
        self.step_size = step_size

        spec_min = torch.FloatTensor(spec_min)[None, None, :out_dims].transpose(-3, -2)
        spec_max = torch.FloatTensor(spec_max)[None, None, :out_dims].transpose(-3, -2)
        self.register_buffer('spec_min', spec_min, persistent=False)
        self.register_buffer('spec_max', spec_max, persistent=False)

    def p_losses(self, x_end, t, cond):
        x_start = torch.randn_like(x_end)
        # sample probability path
        path_sample = self.path.sample(t=t, x_0=x_start, x_1=x_end)
        v_pred = self.denoise_fn(path_sample.x_t, path_sample.t, cond)

        return v_pred, path_sample.dx_t

    def forward(self, cond, gt_spec=None, infer_step=10, infer=True):
        cond = cond.transpose(1, 2)
        b, device = cond.shape[0], cond.device
        if not infer:
            spec = self.norm_spec(gt_spec).transpose(-2, -1)  # [B, M, T] or [B, F, M, T]
            if self.num_features == 1:
                spec = spec[:, None, :, :]
            t = torch.rand((b,), device=device)
            v_pred, v_gt = self.p_losses(spec, t, cond=cond)
            return v_pred, v_gt, t
        else:
            x = self.inference(cond, b=b, infer_step=infer_step, device=device)
            return self.denorm_spec(x)
    
    class WrappedModel(ModelWrapper):
        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
            b, device = x.shape[0], x.device
            t = torch.Tensor([t], device=device).repeat(b)
            return self.model(x, t, extras['cond'])

    @torch.no_grad()
    def inference(self, cond, b=1, infer_step=20, device=None):
        if not hasattr(self, "solver"):
            self.solver = ODESolver(velocity_model=self.WrappedModel(self.denoise_fn))
        x = torch.randn(b, self.num_features, self.out_dims, cond.shape[2], device=device)
        T = torch.linspace(0, 1, infer_step, device=device)
        x = self.solver.sample(time_grid=T, x_init=x, step_size=self.step_size, cond=cond)
        x = x.float()
        x = x.transpose(2, 3).squeeze(1)  # [B, F, M, T] => [B, T, M] or [B, F, T, M]
        return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min