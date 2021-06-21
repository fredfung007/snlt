# COPYRIGHT 2021. Fred Fung. Boston University.

from absl import logging


def check_keys(model, pretrained_state_dict, rank):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    missing_keys = [x for x in missing_keys if not x.endswith('num_batches_tracked')]

    if rank == 0:
        if len(missing_keys) > 0:
            logging.warning('MISSING KEYS: {}'.format(missing_keys))
            logging.warning('MISSING KEYS: {}'.format(len(missing_keys)))
        if len(unused_pretrained_keys) > 0:
            logging.warning('UNUSED_PRETRAINED_KEYS: {}'.format(unused_pretrained_keys))
            logging.warning('UNUSED CHECKPOINT KEYS: {}'.format(len(unused_pretrained_keys)))
            input('CONFIRM CONTINUE...')
        logging.info('USED KEYS: {}'.format(len(used_pretrained_keys)))
        if len(used_pretrained_keys) == 0:
            logging.error('LOAD NONE FROM PRE-TRAINED CHECKPOINT')
            input('CONFIRM CONTINUE...')
    return True
