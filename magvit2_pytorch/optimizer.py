from torch.optim import AdamW, Adam

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, no_wd_params

def get_optimizer(
    params,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    filter_by_requires_grad = False,
    group_wd_params = True,
    **kwargs
):
    if filter_by_requires_grad:
        params = [t for t in params if t.requires_grad]

    opt_kwargs = dict(lr = lr, betas = betas, eps = eps)

    if wd == 0:
        return Adam(params, **opt_kwargs)

    opt_kwargs = {'weight_decay': wd, **opt_kwargs}

    if not group_wd_params:
        return AdamW(params, **opt_kwargs)

    wd_params, no_wd_params = separate_weight_decayable_params(params)

    params = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(params, **opt_kwargs)
