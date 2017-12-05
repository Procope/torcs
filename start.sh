#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver

if __name__ == '__main__':
    main(MyDriver(logdata=False, network_file='single_driver/network_winner.pickle'))
