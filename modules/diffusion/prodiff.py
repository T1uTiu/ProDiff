from functools import partial
from inspect import isfunction
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def vpsde_beta_t(t, T, min_beta, max_beta):
    t_coef = (2 * t - 1) / (T ** 2)
    return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)

def logsnr_schedule_cosine(t, *, logsnr_min, logsnr_max):
  b = np.arctan(np.exp(-0.5 * logsnr_max))
  a = np.arctan(np.exp(-0.5 * logsnr_min)) - b
  return -2. * np.log(np.tan(a * t + b))

def get_noise_schedule_list(schedule_mode, timesteps, min_beta=0.0, max_beta=0.01, s=0.008):
    if schedule_mode == "linear":
        schedule_list = np.linspace(1e-4, max_beta, timesteps)
    elif schedule_mode == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        schedule_list = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_mode == "vpsde":
        schedule_list = np.array([
            vpsde_beta_t(t, timesteps, min_beta, max_beta) for t in range(1, timesteps + 1)])
    elif schedule_mode == "logsnr":
        u = np.array([t for t in range(0, timesteps + 1)])
        schedule_list = np.array([
            logsnr_schedule_cosine(t / timesteps, logsnr_min=-20.0, logsnr_max=20.0) for t in range(1, timesteps + 1)])
    else:
        raise NotImplementedError
    return schedule_list

class GaussianDiffusion(nn.Module):
    def __init__(self, out_dims, denoise_fn,
                 timesteps=1000, time_scale=1, num_features=1,
                 betas=None, schedule_type="vpsde", max_beta=0.02,
                 spec_min=None, spec_max=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.mel_bins = out_dims
        self.num_features = num_features

        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = get_noise_schedule_list(
                schedule_mode=schedule_type,
                timesteps=timesteps + 1,
                min_beta=0.1,
                max_beta=max_beta,
                s=0.008,
            )

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.time_scale = time_scale
        self.num_timesteps = int(timesteps)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('timesteps', to_torch(self.num_timesteps))      # beta
        self.register_buffer('timescale', to_torch(self.time_scale))      # beta
        self.register_buffer('betas', to_torch(betas))      # beta
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod)) # alphacum_t
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev)) # alphacum_{t-1}

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        spec_min = torch.FloatTensor(spec_min)[None, None, :out_dims].transpose(-3, -2)
        spec_max = torch.FloatTensor(spec_max)[None, None, :out_dims].transpose(-3, -2)
        self.register_buffer('spec_min', spec_min, persistent=False)
        self.register_buffer('spec_max', spec_max, persistent=False)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_sample(self, x_start, x_t, t):
        b, *_, device = *x_start.shape, x_start.device
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        noise = torch.randn(x_start.shape, device=device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_start.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        x_0_pred = self.denoise_fn(x_t, t, cond)
        return self.q_posterior_sample(x_start=x_0_pred, x_t=x_t, t=t)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


    def forward(self, cond, gt_spec=None, infer_step=4, infer=False):
        b, *_, device = *cond.shape, cond.device
        cond = cond.transpose(1, 2)
        if not infer:
            spec = self.norm_spec(gt_spec).transpose(-2, -1)  # [B, M, T] or [B, F, M, T]
            if self.num_features == 1:
                spec = spec[:, None, :, :]
            t = torch.randint(0, self.num_timesteps + 1, (b,), device=device).long()
            x_t = self.q_sample(spec, t=t)
            x_0_pred = self.denoise_fn(x_t, t, cond)
            return x_0_pred, spec, t
        else:
            infer_step = np.clip(infer_step, 1, self.num_timesteps)
            x = torch.randn(b, self.num_features, self.mel_bins, cond.shape[2], device=device)  # noise
            for i in tqdm(range(infer_step-1, -1, -1), desc='Sample time step', total=infer_step):
                t = torch.full((b,), i, device=device, dtype=torch.long)
                x = self.p_sample(x, t, cond)  # x(mel), t, condition(phoneme)
            x = x[:, 0].transpose(1, 2)
            x_0 = self.denorm_spec(x)  # 去除norm
        return x_0

    def norm_spec(self, x):
        return x

    def denorm_spec(self, x):
        return x
    
class MultiVariDiffusion(GaussianDiffusion):
    def __init__(self, repeat_bins, denoise_fn,
                 timesteps=1000, time_scale=1,
                 betas=None, schedule_type="vpsde",
                 spec_min: list=None, spec_max: list=None,
                 clamp_min: list=None, clamp_max: list=None):
        assert len(spec_min) == len(spec_max) == len(clamp_min) == len(clamp_max)
        self.num_features = len(spec_min)
        self.clamp_min, self.clamp_max = clamp_min, clamp_max
        self.repeat_bins = repeat_bins
        spec_min = [[v] for v in spec_min]
        spec_max = [[v] for v in spec_max]
        super().__init__(
            out_dims=repeat_bins, denoise_fn=denoise_fn,
            timesteps=timesteps, time_scale=time_scale, num_features=self.num_features,
            betas=betas, schedule_type=schedule_type,
            spec_min=spec_min, spec_max=spec_max
        )

    def clamp_spec(self, xs: list):
        clamped = []
        for x, cmin, cmax in zip(xs, self.clamp_min, self.clamp_max):
            if cmin is not None and cmax is not None:
                clamped.append(x.clamp(cmin, cmax))
            else:
                clamped.append(x)
        xs = torch.stack(clamped, dim=1)
        return xs

    def norm_spec(self, xs: list):
        """
        :param xs: sequence of [B, T]
        :return: [B, F, T, R] or [B, T, R]
        """
        xs = self.clamp_spec(xs)
        repeats = [1, 1, 1, self.repeat_bins]
        if self.num_features == 1:
            xs = xs.squeeze(1)
            repeats = [1, 1, self.repeat_bins]
        return super().norm_spec(xs.unsqueeze(-1).repeat(*repeats))
    
    def denorm_spec(self, xs):
        """
        :param xs: [B, T, R] or [B, F, T, R] => mean => [B, T] or [B, F, T]
        :return: sequence of [B, T]
        """
        denorm_out = super().denorm_spec(xs).mean(dim=-1)
        if self.num_features == 1:
            denorm_out = [denorm_out]
        else:
            denorm_out = denorm_out.unbind(1)
        return self.clamp_spec(denorm_out)

