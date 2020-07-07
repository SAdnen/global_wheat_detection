from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau



def _cosineannealingwarmrestarts(otpimizer, params):

    """
    Example of use
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                            T_0=1,
                            T_mult=2,
                            eta_min=1e-7,
                            last_epoch=-1)
    scheduler.step(e + b / ldloader)
    """
    T_0 = params["T_0"]
    T_mult = params["T_mult"]
    eta_min = params["eta_min"]
    return CosineAnnealingWarmRestarts(optimizer=otpimizer,
                                       T_0=T_0,
                                       T_mult=T_mult,
                                       eta_min=eta_min)


def _onecyclelr(optimizer, params):
    max_lr = params["max_lr"]
    epochs = params["epochs"]
    steps_pre_epoch = params["steps_per_epoch"]
    div_factor = params["div_factor"]
    final_div_factor = params["final_div_factor"]
    return OneCycleLR(optimizer=optimizer,
                      max_lr=max_lr,
                      epochs=epochs,
                      steps_per_epoch=steps_pre_epoch,
                      div_factor=div_factor,
                      final_div_factor=final_div_factor)


def _cosine_annealing_warm_restarts(optimizer, params):
    T_0 = params["T_0"]  # len(dloader) + 1
    T_mult = 2
    eta_min = params["eta_min"]  # min lr
    last_epoch = -1
    return CosineAnnealingWarmRestarts(optimizer=optimizer,
                                       T_0=T_0,
                                       T_mult=T_mult,
                                       eta_min=eta_min,
                                       last_epoch=last_epoch
                                       )

class Scheduler(object):
    schedulers = {"OneCycleLR": _onecyclelr,
                  "CosineAnnealingWarmRestarts": _cosine_annealing_warm_restarts,
                  "ReduceLROnPlateau": ReduceLROnPlateau}

    def get_lr_scheduler(self, optimizer, cfg):
        reference = cfg["lr_scheduler"]
        scheduler_params = cfg["lr_scheduler_params"]
        available_schedulers = ", ".join(self.schedulers.keys())
        assert (reference in self.schedulers.keys()), "Please choose one of: [{}].".format(available_schedulers)
        return self.schedulers[reference](optimizer, scheduler_params)
