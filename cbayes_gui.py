#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2014 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Look at the output of a cbayes analysis using the browser.
"""
import os
import argparse

import cherrypy
from mako.template import Template
from mako.lookup import TemplateLookup

import webbrowser

import numpy as np
import matplotlib
import matplotlib.cm as cm
# use Agg backend to avoid multithread issue with matplotlib
# -- this must be called before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cmpy
from cmpy.math import log, logaddexp
import cmpy.inference.bayesianem as bayesem
import cmpy.orderlygen.pyicdfa as pyidcdfa

import cbayes
import datautils

# set lookup directory for templates
filepath = os.path.dirname(os.path.realpath(__file__))
lookup = TemplateLookup(directories=[os.path.join(filepath,'html')])

def create_parser():
    """Create argparse instance for script.
    
    Returns
    -------
    args : agparse arg instance

    """
    desc_str = (
        """Look at the results of inference with cbayes scripts."""
        )

    parser = argparse.ArgumentParser(description=desc_str)
    
    parser.add_argument('-dir', '--directory',
            help = 'name of the cbayes ouput directory',
            type = str,
            required = True
            )
    
    # do the parsing
    args = parser.parse_args()

    return args

def get_model_evidence(focusdir):
    # get the evidece from the requested directry
    modelevidence = cbayes.read_evidence_file(focusdir)

    return modelevidence
        
def get_model_probs(focusdir, beta):
    # initialize
    modelprobs = {}
    
    # create start of filename for this beta
    fname_start = "probabilities_beta-{:.6f}".format(beta)

    # get the probabilites from the requested directry
    if os.path.isdir(focusdir):
        files = os.listdir(focusdir)

        filename = ''
        modelprobs = {}
        for currf in files:
            if currf.startswith(fname_start):
                filename = os.path.join(focusdir, currf)
                modelprobs = cbayes.read_probabilities_file(filename)

    return modelprobs

def get_top_models(basedir, prob_dirs, beta, nM=3):
    """Generate a tuple containing the top nM (=3 by default) models for all
    inferEM directory.
    
    """
    # initialize
    topmodels = []
    # process rest of data regions for same models
    for inferdir in sorted(prob_dirs.keys()):
        # set focus directory
        focusdir = os.path.join(basedir, prob_dirs[inferdir])
        # obtain model probabilities
        modelprobs = get_model_probs(focusdir, float(beta))
        
        for em in sorted(modelprobs, key=modelprobs.get, reverse=True)[0:nM]:
            if em not in topmodels:
                # add model if not already in list
                topmodels.append(em)
    
    return topmodels

class Root(object):
    
    def __init__(self, args):
        """Initialize the start/root page"""

        self.directory = args.directory
        
        # generate string with only base data directory
        dirstring = os.path.basename(os.path.normpath(self.directory))
        self.string_directory = "Data: {}".format(dirstring) 

        # get location of this file
        self.filepath = os.path.dirname(os.path.realpath(__file__))

        # initialize attributes to hold files and (inferEM, sample) dirs
        self.files = []
        self.betas = []
        self.compare_segs_dict = {}
        self.sample_dirs = {}
        self.prob_dirs = {}
        self.full_dataset = None
        self.full_dataset_tuple = None
    
    @cherrypy.expose
    def compare_segments(self, beta, nM=3):
        # make sure strings are caste
        beta = float(beta)
        nM = int(nM)
        
        # check that it hasn't already been genereated
        plotname = os.path.join(self.filepath,
                'tmp/segments_barplot-beta_{:.6f}-nM_{}.png'.format(beta, nM))

        # for cherrypy retrival
        plotname_short = 'segments_barplot-beta_{:.6f}-nM_{}.png'.format(beta, nM)

        if os.path.exists(plotname):
            # format and return
            image_tmpl = lookup.get_template("singleimage.html.mako")
            tmpl = lookup.get_template("main.html.mako")
            return tmpl.render(title="cbayes inspector",
                               header=self.string_directory,
                               navigation=self.compare_segs_dict,
                               footer="FOOTER",
                               content = image_tmpl.render(filename=plotname_short,
                                 title="Model Probs for Each Data Segment")
                               )
        
        # for proper sorting of segments
        def segment_keys(args):
            seg1, seg2 = args
            return (seg1,-seg2)

        segment_labels = sorted(self.prob_dirs.keys(), key=segment_keys)
        number_dirs = len(segment_labels)
        
        # get top nM models for all inferEM directories
        topmodels = get_top_models(self.directory, self.prob_dirs, beta, nM)
        number_models = len(topmodels)

        # build data
        data = []
        baseline = []
        for n, em in enumerate(topmodels):
            data.append([0 for t in range(number_dirs)])
            baseline.append( [0 for t in range(number_dirs)])

        # populate fields
        for n, em in enumerate(topmodels):
            for s, seg in enumerate(segment_labels):
                focusdir = os.path.join(self.directory, self.prob_dirs[seg])
                modelprobs = get_model_probs(focusdir, beta)

                data[n][s] = modelprobs[em]
                if n == 0:
                    baseline[n][s] = 0.
                else:
                    baseline[n][s] = baseline[n-1][s] + data[n-1][s]

        #import matplotlib.cm as cm
        from matplotlib.font_manager import FontProperties
        fontP = FontProperties()
        fontP.set_size('small')
        
        # define colormap
        colors = iter(cm.terrain(np.linspace(0, 1, number_models)))
        
        # make stacked bar plot
        width = 1.0      # the width of the bars: can also be len(x) sequence
        ind = np.arange(number_dirs) # [n for n in range(number_dirs)]
       
        fig = plt.figure()
        ax  = fig.add_subplot(111)

        plots = []
        for n, em in enumerate(topmodels):
            plots.append(ax.bar(ind + width/2., data[n], bottom=baseline[n],
                                 color=next(colors)))
        
        plt.ylabel('Probabilities')
        plt.xticks(ind +width/2., segment_labels, rotation=45)
        plt.yticks([0., 0.25, 0.5, 0.75, 1.0])
        colors = [p[0] for p in plots]
        # modify axis to fit legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1.25, box.height])
        
        # create legend
        ax.legend(tuple(colors), topmodels, loc='center left',
                bbox_to_anchor=(1, 0.5), prop=fontP)
        # make sure bounding box is good
        plt.tight_layout()
        
        plt.savefig(plotname)

        # clear plot
        plt.clf()
        
        # format and return
        image_tmpl = lookup.get_template("singleimage.html.mako")
        tmpl = lookup.get_template("main.html.mako") 
        return tmpl.render(title="cbayes inspector",
                           header=self.string_directory,
                           navigation=self.compare_segs_dict,
                           footer="FOOTER",
                           content = image_tmpl.render(filename=plotname_short,
                             title="Model Probs for Each Data Segment")
                           )

    @cherrypy.expose
    def index(self):
        abs_min = 0
        abs_max = 0
        # iterate through contents of specified directory
        for f in os.listdir(self.directory):
            if os.path.isdir(os.path.join(self.directory,f)):
                # get data range from dir name
                data_range = f.split('_')[1]
                data_low, data_high = data_range.split('-')
                data_low = int(data_low)
                data_high = int(data_high)

                # keep track of lowest and highest range
                if abs_min > data_low:
                    abs_min = data_low
                if abs_max < data_high:
                    abs_max = data_high

                # process by dir type
                if f.startswith('inferEM'):
                    # is a dir with log_evidence and probabilities files
                    data_tuple = (data_low, data_high)
                    # save dir name
                    self.prob_dirs[data_tuple] = f
                elif f.startswith('sample'):
                    # is a dir with sample file
                    beta_info = f.split('_')[2]
                    beta = float(beta_info.split('-')[1])
                    # accumulate list of betas
                    if beta not in self.betas:
                        self.betas.append(beta)
                    # create identifying tuple
                    data_tuple = (data_low, data_high, beta)
                    # save dir name
                    self.sample_dirs[data_tuple] = f
            elif os.path.isfile(os.path.join(self.directory,f)):
                self.files.append(f)
        
        # save name of full dataset directory
        self.full_dataset_tuple = (abs_min, abs_max)
        self.full_dataset = self.prob_dirs[self.full_dataset_tuple]

        # construct compars segments dictionary for header
        for beta in self.betas:
            key = "CS<br>beta={:.2f}".format(beta)
            val = "/compare_segments/{:f}".format(beta)
            self.compare_segs_dict[key] = val

        # format page content
        page_tmpl = lookup.get_template("files.table.html.mako")
        
        # format and return
        tmpl = lookup.get_template("main.html.mako")
        return tmpl.render(title="cbayes inspector",
                           header=self.string_directory,
                           navigation=self.compare_segs_dict,
                           footer="FOOTER",
                           content = page_tmpl.render(prob_dirs=self.prob_dirs,
                                                      sample_dirs=self.sample_dirs,
                                                      betas=self.betas)
                           )
    
    @cherrypy.expose
    def mapmachine(self, inferdir, emtop):
        focusdir = os.path.join(self.directory, inferdir)
        
        # create new InferEM instance
        # - number of states
        n = int(emtop.split('_')[0][1:])
        # - alphabet size
        k =  int(emtop.split('_')[1][1:])
        # - id
        id =  int(emtop.split('_')[2][2:])
        
        # get machine topology
        machine = pyidcdfa.int_to_machine(id, n, k)
        # set name, with proper ID - n states, k symbols
        mname = "n{}_k{}_id{}".format(n, k, id)
        machine.set_name(mname)

        # generate inferEM instance
        data = cbayes.read_datafile(os.path.join(self.directory, 'datafile'))
        inferem = bayesem.InferEM(machine, data)
        pm_machines = inferem.get_PM_machines()
        
        # get startnode probabilities
        snprobs = {}
        for sn in inferem.get_possible_start_nodes():
            snprobs[sn] = inferem.probability_start_node(sn)

        machinedict = {}
        for startnode in pm_machines:
            # make filename
            fname = '{}_sn{}_{}.png'.format(inferdir, startnode, mname)
            
            # draw machine
            em = pm_machines[startnode]
            plotfilename = os.path.join(self.filepath,
                    "tmp/{}".format(fname))
            if not os.path.exists(plotfilename):
                # graphic does not exist, draw
                em.draw(filename=plotfilename, format='png', show=False)
            else:
                pass


            # add machine
            machinedict[startnode] = fname
        
        # get image gallery template
        image_tmpl = lookup.get_template("imagegallery.html.mako")

        tmpl = lookup.get_template("main.html.mako")
        return tmpl.render(title=mname,
                           header=self.string_directory,
                           navigation=self.compare_segs_dict,
                           content = image_tmpl.render(imagedict=machinedict,
                                                       sndict=snprobs),
                           footer="FOOTER")

    @cherrypy.expose
    def mprobs(self, inferdir, beta, nM):
        # set focus directory
        focusdir = os.path.join(self.directory, inferdir)
        # obtain model probabilities
        modelprobs = get_model_probs(focusdir, float(beta))
        # obtain model evidence values
        modelevidence = get_model_evidence(focusdir)

        # proces nM
        dictsize = len(modelprobs.keys())
        if nM == 'all':
            nM = 100
        elif int(nM) > dictsize:
            nM = 100
        else:
            nM = int(nM)
        
        # massage the data for mako
        modelinfo = []
        cpr = 0.
        tdict = dict(modelprobs)
        for em in sorted(tdict, key=tdict.get, reverse=True)[0:nM]:
            cpr += tdict[em]
            modelinfo.append((em, tdict[em], cpr, modelevidence[em]))
        
        # format and return
        tmpl = lookup.get_template("main.html.mako") 
        tmpl_table = lookup.get_template("prob.table.html.mako")
        return tmpl.render(title="cbayes inspector",
                           header=self.string_directory,
                           navigation=self.compare_segs_dict,
                           footer="FOOTER",
                           content = tmpl_table.render(mi=modelinfo,
                                                       nM=10,
                                                       inferdir=inferdir)
                           )

    @cherrypy.expose
    def hmuCmu_sampleplot(self, sampledir):
        focusdir = os.path.join(self.directory, sampledir)
        
        # create plotname
        plotname = os.path.join(self.filepath, 'tmp/{}.png'.format(sampledir))
        # for cherrypy
        plotname_short = '{}.png'.format(sampledir)

        # check that it hasn't already been genereated
        if os.path.exists(plotname):
            # format and return
            image_tmpl = lookup.get_template("singleimage.html.mako")
            tmpl = lookup.get_template("main.html.mako") 
            return tmpl.render(title="cbayes inspector",
                               header=self.string_directory,
                               navigation=self.compare_segs_dict,
                               footer="FOOTER",
                               content = image_tmpl.render(filename=plotname_short,
                                   title='hmu-Cmu samples')
                               )

        
        # get samples from the requested directry
        if os.path.isdir(focusdir):
            files = os.listdir(focusdir)

            filename = ''
            modelprobs = {}
            for currf in files:
                if currf.startswith('samples'):
                    filename = os.path.join(focusdir, currf)
                    samples = datautils.FileSlice(filename, ',')
        
        # make plot
        from matplotlib.ticker import NullFormatter
        
        x = samples.get_data_column('hmu')
        xmin = min(x)
        xmax = max(x)
        # check for xmin=xmax
        if xmin == xmax:
            xmin -= 0.1
            xmax ++ 0.1

        y = samples.get_data_column('Cmu')
        ymin = min(y)
        ymax = max(y)
        # check for ymin=ymax
        if ymin == ymax:
            ymin -= 0.1
            ymax += 0.1
        
        nullfmt   = NullFormatter()         # no labels
        
        # definitions for the axes 
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width+0.02
        
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        
        # start with a rectangular Figure
        plt.figure(1, figsize=(8,8))
        
        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        
        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        
        # the scatter plot:
        axScatter.scatter(x, y)
        scat_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        axScatter.yaxis.set_major_formatter(scat_formatter)
        axScatter.xaxis.set_major_formatter(scat_formatter)
        
        xlow  = xmin - 0.1*(xmax-xmin)
        xhigh = xmax + 0.1*(xmax-xmin)
        ylow  = ymin - 0.1*(ymax-ymin)
        yhigh = ymax + 0.1*(ymax-ymin)
        
        axScatter.set_xlim((xlow, xhigh))
        axScatter.set_ylim((ylow, yhigh))
        axScatter.set_ylabel('Cmu [bits]')
        axScatter.set_xlabel('hmu [bits]')
        
        dx = (xmax-xmin)/50
        xbins = np.arange(xlow, xhigh+dx, dx)
        axHistx.hist(x, bins=xbins, normed=False)
        
        # move tick labels to left on Cmu histogram
        dy = (ymax-ymin)/50
        ybins = np.arange(ylow, yhigh+dy, dy)
        axHisty.hist(y, bins=ybins, normed=False, orientation='horizontal')

        axHistx.set_xlim( axScatter.get_xlim() )
        axHisty.set_ylim( axScatter.get_ylim() )
        
        plt.savefig(plotname)

        # delete plot
        plt.clf()
        
        # format and return
        image_tmpl = lookup.get_template("singleimage.html.mako")
        tmpl = lookup.get_template("main.html.mako") 
        return tmpl.render(title="cbayes inspector",
                           header=self.string_directory,
                           navigation=self.compare_segs_dict,
                           footer="FOOTER",
                           content = image_tmpl.render(filename=plotname_short,
                            title="hmu-Cmu samples")
                           )

    @cherrypy.expose
    def shutdown(self):
        # do some cleanup in the tmp/ directory
        try:
            os.chdir(os.path.join(self.filepath, 'tmp'))
            # get all png's
            filelist = [ f for f in os.listdir(".") if f.endswith(".png") ]
            # remove
            for f in filelist:
                os.remove(f)
        except:
            pass

        # shutdown server
        cherrypy.engine.exit()

if __name__ == "__main__":
    # get command line args
    args = create_parser()

    # get path to this file
    filepath = os.path.dirname(os.path.realpath(__file__))
    
    # open browser at localhost:8080
    webbrowser.open('http://localhost:8080/', new=1)
    
    # config and startup cherrypy
    cherrypy.server.socket_host = "127.0.0.1"
    cherrypy.server.socket_port = 8080
    config = {
        "/": {
            "tools.staticdir.root": filepath,
        },
        "/html": {
            "tools.staticdir.on": True,
            "tools.staticdir.dir": 'tmp',
        },
        "/tmp": {
            "tools.staticdir.on": True,
            "tools.staticdir.dir": 'tmp',
        },
        "/css": {
            "tools.staticdir.on": True,
            "tools.staticdir.dir": 'css',
        }
    }
    cherrypy.tree.mount(Root(args), "/", config=config)
    cherrypy.engine.start()
    cherrypy.engine.block()

