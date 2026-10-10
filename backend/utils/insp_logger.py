import logging
import os
import datetime 
formatter = logging.Formatter('%(asctime)s.%(msecs)d %(message)s', '%Y-%m-%d %H:%M:%S')
def setup_logger(name, log_file, level=logging.INFO, console_output=True):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    # logger.addHandler(console_handler)

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

today = datetime.date.today()
c = datetime.datetime.now()
current_time = c.strftime('%H-%M-%S')

log_path =  f'logs/{today}_{current_time}/'
os.makedirs(f'logs/{today}_{current_time}/', exist_ok=True)

user_log_file = f'avery_user_log_{current_time}.log'
user_log_name = os.path.join(log_path, user_log_file)
user_logger = setup_logger('avery_user', user_log_name)

dev_log_file = f'avery_dev_log_{current_time}.log'
dev_log_name = os.path.join(log_path, dev_log_file)
dev_logger = setup_logger('avery_dev', dev_log_name, console_output=False)