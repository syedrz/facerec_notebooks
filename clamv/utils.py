import logging
import logstash
from clamv import config
import boto3
import cv2

s3 = boto3.resource('s3')


def get_logger(package_name):
    debug_log = logging.FileHandler('debug.log')
    info_log = logging.FileHandler('info.log')

    logstash_log = logstash.TCPLogstashHandler(config.LOGSTASH_HOST,
                                               config.LOGSTASH_PORT,
                                               version=1)

    debug_log.setLevel(logging.DEBUG)
    info_log.setLevel(logging.INFO)
    logstash_log.setLevel(logging.DEBUG)

    logger = logging.getLogger(package_name)
    logger.setLevel(logging.DEBUG)

    logger.addHandler(debug_log)
    logger.addHandler(info_log)
    logger.addHandler(logstash_log)

    return logger


def s3_put(key, image):
    ret, binary = cv2.imencode('.jpg', image)
    if ret:
        binary = binary.flatten().tobytes()
        with open(f'out/{key}', 'wb') as f:
            f.write(binary)
        s3.Bucket(config.S3_BUCKET).put_object(Key=key, Body=binary)
    else:
        raise Exception('Could not encode image')
