import random


class RandomSelector:

    def select_next_batch(self, trainer, active_set, selection_count):
        scores = []
        for i in range(len(active_set.trg_pool_dataset.im_idx)):
            scores.append(random.random())
        selected_samples = list(zip(*sorted(zip(scores, active_set.trg_pool_dataset.im_idx),
                                            key=lambda x: x[0], reverse=True)))[1][:selection_count]
        active_set.expand_training_set(selected_samples)


class RegionRandomSelector:

    def select_next_batch(self, trainer, active_set, selection_count):
        if trainer.local_rank == 0:
            scores = []
            # Give each superpixel a random score
            for key in active_set.trg_pool_dataset.im_idx:
                rgb_fname, gt_fname, spx_fname = key
                for suppix_id in active_set.trg_pool_dataset.suppix[spx_fname]:
                    score = random.random()
                    file_path = ",".join(key)
                    item = (score, file_path, suppix_id)
                    scores.append(item)
            # Sort the score
            selected_samples = sorted(scores, reverse=True)
            active_set.expand_training_set(selected_samples, selection_count)
