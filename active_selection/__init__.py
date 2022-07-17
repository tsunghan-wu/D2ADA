from active_selection import softmax_uncertainty
from active_selection import density_aware_selection


def get_density_selector(FLAGS):
    return density_aware_selection.RegionDensitySelector(FLAGS, 3, 3)


def get_uncertain_selector(FLAGS):
    return softmax_uncertainty.RegionSoftmaxUncertaintySelector(4, 4, 'softmax_entropy')


'''
    Other implemented active learning baselines.
    If you are interested in active learning and would like to conduct further research,
    feel free to use or modify these code and cite our paper =)

def get_active_selector(FLAGS):
    if FLAGS.active_method == 'random':
        return random_selection.RegionRandomSelector()
    elif FLAGS.active_method == 'softmax_confidence':
        return softmax_uncertainty.RegionSoftmaxUncertaintySelector(4, 4, 'softmax_confidence')
    elif FLAGS.active_method == 'softmax_margin':
        return softmax_uncertainty.RegionSoftmaxUncertaintySelector(4, 4, 'softmax_margin')
    elif FLAGS.active_method == 'softmax_entropy':
        return softmax_uncertainty.RegionSoftmaxUncertaintySelector(4, 4, 'softmax_entropy')
    elif FLAGS.active_method == 'gmm':
        return GMM_selection.RegionGMMSelector(FLAGS, 2, 2)
    elif FLAGS.active_method == 'clue':
        return clue.CLUESelector(FLAGS, 2, 2)
    elif FLAGS.active_method == 'aada':
        return AADA.RegionAADASelector(4, 4)
    elif FLAGS.active_method == 'ReDAL':
        return ReDAL.ReDALSelector(2, 2, 500, 0.95)
    elif FLAGS.active_method == 'BADGE':
        return badge.BADGESelector(FLAGS, 3, 3)
'''
