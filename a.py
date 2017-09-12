#!/usr/bin/env python
import getpass
import logging
import logging.handlers
import os
import random
import sys

import chatexchange.client
import chatexchange.events

from subprocess import list2cmdline

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    # Run `. setp.sh` to set the below testing environment variables

    host_id = 'stackexchange.com'
    room_id = '11540'  # Sandbox

    email = "deepsmokey@gmail.com"
    password = "wwjNgn99!@"

    client = chatexchange.client.Client(host_id)
    client.login(email, password)

    room = client.get_room(room_id)
    room.join()
    #room.watch(on_message)

    room.send_message(sys.argv[1])

    print("(You are now in room #%s on %s.)" % (room_id, host_id))
    while True:
	i = 0

    client.logout()


def on_message(message, client):
    if not isinstance(message, chatexchange.events.MessagePosted):
        # Ignore non-message_posted events.
        logger.debug("event: %r", message)
        return

    print("")
    print(">> (%s) %s" % (message.user.name, message.content))
    if message.content.startswith('!DeepSmokey'):
        print(message)
        print("Spawning thread")
        finalmessage = message.content.replace("!DeepSmokey ", "")
        unicodeStripped = "".join([i if ord(i) < 128 else '' for i in finalmessage])
        list4cmdline = ["python", "getresult.py", unicodeStripped]
        osResult = os.popen(list2cmdline(list4cmdline)).read()
        message.message.reply(osResult)
    elif message.content.startswith('[ SmokeDetector | MS ]'):
        regex = '/\[ <a[^>]+>SmokeDetector<\/a>(?: \| <a[^>]+>MS<\/a>)? ] ([^:]+):(?: post \d+ out of \d+\):)? <a href="([^"]+)">(.+?)<\/a> by (?:<a href="[^"]+\/u\/(\d+)">(.+?)<\/a>|a deleted user) on <code>([^<]+)<\/code>/'
        print("Detected SD post")

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)

    # In addition to the basic stderr logging configured globally
    # above, we'll use a log file for chatexchange.client.
    wrapper_logger = logging.getLogger('chatexchange.client')
    wrapper_handler = logging.handlers.TimedRotatingFileHandler(
        filename='client.log',
        when='midnight', delay=True, utc=True, backupCount=7,
    )
    wrapper_handler.setFormatter(logging.Formatter(
        "%(asctime)s: %(levelname)s: %(threadName)s: %(message)s"
    ))
    wrapper_logger.addHandler(wrapper_handler)


if __name__ == '__main__':
    main()
