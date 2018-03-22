#!usr/bin/python

import sys
import ConfigParser
import os

class GetConfigs:

    def __init__(self, config):
        self.config = config

    def get_sections(self):
        config = ConfigParser.SafeConfigParser()
        config.read(self.config)
        sections_list = config.sections()
        return sections_list

    def get_options(self, key):
        config = ConfigParser.SafeConfigParser()
        config.read(self.config)
        section_opts = config.options(key)
        return section_opts

    def get_optitem(self,key):
        config = ConfigParser.SafeConfigParser()
        config.read(self.config)
        options_item = config.items(key)
        return dict(options_item)


if __name__== '__main__':

    if len(sys.argv) == 1:
        print 'Must provide the path to the config file as the argument'
        sys.exit(1)

    # reading the .ini file
    getconfig = GetConfigs(sys.argv[1])

    # getting the sections
    config_sec = getconfig.get_sections()

    # getting the pipeline section options and items
    pipe_opt = getconfig.get_options('pipeline')
    pipe_dict = getconfig.get_optitem('pipeline')

    # starting with vt: if module_vt=n, dont run vt, else ->source, read configs and run vt
    for i in pipe_opt:
        if i=='module_vt':
            run_vt = pipe_dict[i]
            if run_vt=='n':
                print 'You are not running VT'
                exit()
            else:
                vt_source_path =  # source <target_dir>/vt/run/setup-vt.sh
                os.system('source vt_source_path')
                os.system('source ...)
                get the configs to run vt
                os.system('vtfind...')
        elif i=='module_ma':
            run_ma = pipe_dict[i]


'''

    if ma_pix_list=='':
        os.system(python...nike)
    elif:
        os.system(source Memb-assign/start.sh)


    for i in config_sec:
        config_opt=getconfig.get_options(i)
        print config_opt
'''
