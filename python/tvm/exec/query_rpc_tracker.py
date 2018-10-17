"""Tool to query RPC tracker status"""
from __future__ import absolute_import

import logging
import argparse
import os
<<<<<<< HEAD
from ..contrib import rpc
=======
from .. import rpc
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="",
                        help='the hostname of the tracker')
    parser.add_argument('--port', type=int, default=None,
                        help='The port of the PRC')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # default to local host or environment variable
    if not args.host:
        args.host = os.environ.get("TVM_TRACKER_HOST", "localhost")

    if not args.port:
        args.port = int(os.environ.get("TVM_TRACKER_PORT", "9190"))

    conn = rpc.connect_tracker(args.host, args.port)
    # pylint: disable=superfluous-parens
    print("Tracker address %s:%d\n" % (args.host, args.port))
    print("%s" % conn.text_summary())

if __name__ == "__main__":
    main()
