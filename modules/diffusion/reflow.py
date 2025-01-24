import torch
from torch import nn
from tqdm import tqdm

class RectifiedFlow(nn.Module):
    def __init__(self, out_dims, denoise_fn, 
                 time_scale=1000, num_features=1, 
                 sampling_algorithm="euler",
                 spec_min=None, spec_max=None):
        super().__init__()
        self.velocity_fn = denoise_fn
        self.out_dims = out_dims
        self.num_features = num_features
        self.sampling_algorithm = sampling_algorithm
        self.t_start = 0.
        self.time_scale = time_scale

        # spec: [B, T, M] or [B, F, T, M]
        # spec_min and spec_max: [1, 1, M] or [1, 1, F, M] => transpose(-3, -2) => [1, 1, M] or [1, F, 1, M]
        spec_min = torch.FloatTensor(spec_min)[None, None, :out_dims].transpose(-3, -2)
        spec_max = torch.FloatTensor(spec_max)[None, None, :out_dims].transpose(-3, -2)
        self.register_buffer('spec_min', spec_min, persistent=False)
        self.register_buffer('spec_max', spec_max, persistent=False)

    def p_losses(self, x_end, t, cond):
        x_start = torch.randn_like(x_end)
        x_t = x_start + t[:, None, None, None] * (x_end - x_start)
        v_pred = self.velocity_fn(x_t, t * self.time_scale, cond)

        return v_pred, x_end - x_start

    def forward(self, cond, nonpadding=None, gt_spec=None, infer_step=20, infer=True):
        cond = cond.transpose(1, 2)
        b, device = cond.shape[0], cond.device

        if not infer:
            # gt_spec: [B, T, M] or [B, F, T, M]
            spec = self.norm_spec(gt_spec).transpose(-2, -1)  # [B, M, T] or [B, F, M, T]
            if self.num_features == 1:
                spec = spec[:, None, :, :]  # [B, F=1, M, T]
            t = self.t_start + (1.0 - self.t_start) * torch.rand((b,), device=device)
            v_pred, v_gt = self.p_losses(spec, t, cond=cond)
            return v_pred, v_gt, t
        else:
            x = self.inference(cond, b=b, infer_step=infer_step, device=device)
            return self.denorm_spec(x)

    @torch.no_grad()
    def sample_euler(self, x, t, dt, cond):
        x += self.velocity_fn(x, self.time_scale * t, cond) * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk2(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, self.time_scale * (t + 0.5 * dt), cond)
        x += k_2 * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk4(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, self.time_scale * (t + 0.5 * dt), cond)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, self.time_scale * (t + 0.5 * dt), cond)
        k_4 = self.velocity_fn(x + k_3 * dt, self.time_scale * (t + dt), cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk5(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale * t, cond)
        k_2 = self.velocity_fn(x + 0.25 * k_1 * dt, self.time_scale * (t + 0.25 * dt), cond)
        k_3 = self.velocity_fn(x + 0.125 * (k_2 + k_1) * dt, self.time_scale * (t + 0.25 * dt), cond)
        k_4 = self.velocity_fn(x + 0.5 * (-k_2 + 2 * k_3) * dt, self.time_scale * (t + 0.5 * dt), cond)
        k_5 = self.velocity_fn(x + 0.0625 * (3 * k_1 + 9 * k_4) * dt, self.time_scale * (t + 0.75 * dt), cond)
        k_6 = self.velocity_fn(x + (-3 * k_1 + 2 * k_2 + 12 * k_3 - 12 * k_4 + 8 * k_5) * dt / 7,
                               self.time_scale * (t + dt),
                               cond)
        x += (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) * dt / 90
        t += dt
        return x, t

    @torch.no_grad()
    def inference(self, cond, b=1, infer_step=20, device=None):
        x = torch.randn(b, self.num_features, self.out_dims, cond.shape[2], device=device)
        dt = 1.0 / max(1, infer_step) # 1 / 20
        algorithm_fn = {
            'euler': self.sample_euler,
            'rk2': self.sample_rk2,
            'rk4': self.sample_rk4,
            'rk5': self.sample_rk5,
        }.get(self.sampling_algorithm, self.sample_euler)
        dts = torch.tensor([dt]).to(x) # to device
        for i in tqdm(range(infer_step), desc='Sample time step', total=infer_step, leave=False):
            x, _ = algorithm_fn(x, i * dts, dt, cond)
        x = x.float()
        x = x.transpose(2, 3).squeeze(1)  # [B, F, M, T] => [B, T, M] or [B, F, T, M]
        return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class PitchRectifiedFlow(RectifiedFlow):
    def __init__(self, repeat_bins, denoise_fn, 
                 time_scale=1000,
                 sampling_algorithm="euler",
                 spec_min=-8.0, spec_max=8.0,
                 clamp_min=-12.0, clamp_max=12.0,):
        spec_min = [spec_min]
        spec_max = [spec_max]
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.repeat_bins = repeat_bins
        super().__init__(
            out_dims=repeat_bins, denoise_fn=denoise_fn, 
            time_scale=time_scale, num_features=1,
            sampling_algorithm=sampling_algorithm,
            spec_min=spec_min, spec_max=spec_max
        )

    def norm_spec(self, x):
        """

        :param x: [B, T] or [B, F, T]
        :return [B, T, R] or [B, F, T, R]
        """
        x = x.clamp(self.clamp_min, self.clamp_max)
        repeats = [1, 1, self.repeat_bins]
        return super().norm_spec(x.unsqueeze(-1).repeat(repeats))

    def denorm_spec(self, x):
        """

        :param x: [B, T, R] or [B, F, T, R]
        :return [B, T] or [B, F, T]
        """
        return super().denorm_spec(x).mean(dim=-1).clamp(self.clamp_min, self.clamp_max)