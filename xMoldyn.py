#!/usr/bin/python


from Tkinter import *
from ScrolledText import ScrolledText
from tkFont import Font
from tkMessageBox import showinfo, askokcancel
import tkFileDialog
from nMoldyn.gui import *
from nMoldyn.misc import getData, getTypes, getAtoms, saveText, getResidues,\
                         getMethyls, getTokens, getProteinBackbone, \
                         parsePDBReference
from nMoldyn.calc import linearRegression, qTrajectory
import nMoldyn.docs 
from MMTK import *
from MMTK.Trajectory import Trajectory, trajectoryInfo, TrajectorySet
from MMTK.PDB import PDBOutputFile, PDBConfiguration
from Scientific.TkWidgets.TkPlotCanvas import PolyLine, \
                PlotGraphics, PlotCanvas, PolyMarker
from Scientific.TkWidgets.TkVisualizationCanvas import PolyLine3D, \
                VisualizationGraphics, VisualizationCanvas
from Numeric import *
from Scientific.IO.TextFile import TextFile
from Scientific.IO.NetCDF import *
from os import *
from string import *
from tempfile import gettempdir, mktemp
import re, os, string, sys, types
import Numeric


class Command:

    def __init__(self, name, menu, position, entry, required_data,
                 parameters, selections, output_files, switch):
        self.name = name
        self.menu_title = menu
        self.menu_position = position
        self.menu_entry = string.split(entry, '|')
        self.required_data = required_data
        self.selections = selections
        self.output_files = output_files
        self.switch = switch
        self.initial_parameters = parameters
        self.setDefaults()

    def __repr__(self):
        return 'Command("%s")' % self.name

    def setDefaults(self):
        self.parameters = {}
        for p in self.initial_parameters:
            if type(p) == types.TupleType:
                p, default = p
            else:
                default = self._defaults[p]
            self.parameters[p] = default
        files = ()
        for extension, tag, id, label in self.output_files:
            filename = tag + '_%s.' + extension
            files = files + (filename,)
        self.parameters['output_files'] = files

    _defaults = {'weights': 'mass',
                 'projection_vector': 'None',
                 'units_length': 'nm',
                 'units_q': '1/nm',
                 'units_frequency': '1/ps',
                 'differentiation': 'fast',
                 'ft_window': '10',
                 'ar_order': '50',
                 'ar_precision': 'None',
                 'rotation_coefficients': '(0, 0, 0)',
                 'filter_window': ('0', 'None'),
                 'time_steps': '2000',
                 'frequency_points': '2000',
                 'q_vector_set': ('0.:2.:100.', '1.', '50', ''),
                 }

    _order = ['rotation_coefficients', 'ar_order', 'ar_precision',
              'filter_window',
              'q_vector_set', 'projection_vector', 'weights',
              'differentiation', 'ft_window',
              'units_length', 'units_q', 'units_frequency',
              'time_steps', 'frequency_points', 'output_files']

    _string_parameters = ['weights', 'differentiation']

    _weight_types = ['none', 'mass', 'incoherent', 'coherent']

    _length_units = ['nm', '\305']
    _frequency_units = ['1/ps', '1/cm']
    _q_units = ['1/nm', '1/\305']

    def isActive(self, available_data):
        active = 1
        for data in self.required_data:
            if not data in available_data:
                active = 0
        return active

    def run(self):
        window = Toplevel(self.master)
        window.title(self.name)
        window.resizable(width=NO, height=NO)
        window.initial_focus = window
        frame = Frame(window, bd=2, relief='groove')
        frame.pack(side=TOP, fill=BOTH, padx=3, pady=3)

        form_data = []
        for parameter in self._order:
            if self.parameters.has_key(parameter):
                value = self.parameters[parameter]
                f = Frame(frame, bd=2, relief='groove')
                f.pack(side=TOP, fill=BOTH, padx=2, pady=2)
                f = Frame(f, bd=2, relief='flat')
                f.pack(side=TOP, fill=BOTH, padx=2, pady=2)
                if parameter == 'rotation_coefficients':
                    Label(f, text='Rotation coefficients: ', anchor=W) \
                             .pack(side=LEFT)
                    variable = StringVar()
                    variable.set(value)
                    entry = Entry(f, textvariable=variable)
                    entry.pack(side=LEFT)
                    form_data.append((parameter, variable))
                elif parameter == 'projection_vector':
                    Label(f, text='Projection vector: ', anchor=W) \
                             .pack(side=LEFT)
                    variable = StringVar()
                    variable.set(value)
                    entry = Entry(f, textvariable=variable)
                    entry.pack(side=LEFT)
                    form_data.append((parameter, variable))
                elif parameter == 'filter_window':
                    Label(f, text='Filter window from: ', anchor=E) \
                             .grid(row=0, column=0, sticky=E)
                    Label(f, text='to: ', anchor=E) \
                             .grid(row=1, column=0, sticky=E)
                    variables = ()
                    for i in range(2):
                        var = StringVar()
                        var.set(value[i])
                        variables = variables + (var,)
                        entry = Entry(f, textvariable=var)
                        entry.grid(row=i, column=1, sticky=W)
                    form_data.append((parameter, variables))
                elif parameter == 'ar_order':
                    Label(f, text='Model order: ', anchor=W) \
                             .pack(side=LEFT)
                    variable = StringVar()
                    variable.set(value)
                    entry = Entry(f, textvariable=variable)
                    entry.pack(side=LEFT)
                    form_data.append((parameter, variable))
                elif parameter == 'ar_precision':
                    Label(f, text='Extended precision (memory function): ',
                          anchor=W).pack(side=LEFT)
                    variable = StringVar()
                    variable.set(value)
                    entry = Entry(f, textvariable=variable)
                    entry.pack(side=LEFT)
                    form_data.append((parameter, variable))
                elif parameter == 'weights':
                    Label(f, text='Weights: ', anchor=W).grid(column=0,
                                                              row=0, sticky=E)
                    weight_type = StringVar()
                    weight_type.set(value)
                    i = 0
                    for w in self._weight_types:
                        Radiobutton(f, variable=weight_type, value=w, text=w)\
                                       .grid(column=1+i/2, row=i%2, sticky=W)
                        i = i + 1
                    form_data.append((parameter, weight_type))
                elif parameter == 'units_length':
                    Label(f, text='Length units: ', anchor=W)\
                             .grid(row=0, column=0, sticky=NE)
                    f = Frame(f)
                    f.grid(row=0, column=1, sticky=W)
                    unit = StringVar()
                    unit.set(value)
                    i = 0
                    for w in self._length_units:
                        Radiobutton(f, variable=unit, value=w, text=w)\
                                       .grid(column=1, row=i, sticky=W)
                        i = i + 1
                    form_data.append((parameter, unit))
                elif parameter == 'units_frequency':
                    Label(f, text='Frequency units: ', anchor=W)\
                             .grid(row=0, column=0, sticky=NE)
                    unit = StringVar()
                    unit.set(value)
                    i = 0
                    for w in self._frequency_units:
                        Radiobutton(f, variable=unit, value=w, text=w)\
                                       .grid(column=1, row=i, sticky=W)
                        i = i + 1
                    form_data.append((parameter, unit))
                elif parameter == 'units_q':
                    Label(f, text='Q units: ', anchor=W)\
                             .grid(row=0, column=0, sticky=NE)
                    unit = StringVar()
                    unit.set(value)
                    i = 0
                    for w in self._q_units:
                        Radiobutton(f, variable=unit, value=w, text=w)\
                                       .grid(column=1, row=i, sticky=W)
                        i = i + 1
                    form_data.append((parameter, unit))
                elif parameter == 'differentiation':
                    Label(f, text='Differentiation: ', anchor=W)\
                             .grid(row=0, column=0, sticky=NE)
                    scheme = StringVar()
                    scheme.set(value)
                    OptionMenu(f, scheme,
                               'fast','order 2','order 3',
                               'order 4','order 5').grid(row=0,
                                                         column=1, sticky=NW)
                    form_data.append((parameter, scheme))
                elif parameter == 'ft_window':
                    Label(f, text='Window width for FFT: ', anchor=W) \
                             .pack(side=LEFT)
                    variable = StringVar()
                    variable.set(value)
                    entry = Entry(f, textvariable=variable)
                    entry.pack(side=LEFT)
                    Label(f, text='% of trajectory length', anchor=W) \
                             .pack(side=LEFT)
                    form_data.append((parameter, variable))
                elif parameter == 'time_steps':
                    Label(f, text='Time steps in output: ', anchor=W) \
                             .pack(side=LEFT)
                    variable = StringVar()
                    variable.set(value)
                    entry = Entry(f, textvariable=variable)
                    entry.pack(side=LEFT)
                    form_data.append((parameter, variable))
                elif parameter == 'frequency_points':
                    Label(f, text='Points in spectrum: ', anchor=W) \
                             .pack(side=LEFT)
                    variable = StringVar()
                    variable.set(value)
                    entry = Entry(f, textvariable=variable)
                    entry.pack(side=LEFT)
                    form_data.append((parameter, variable))
                elif parameter == 'q_vector_set':
                    Label(f, text='Q values: ',
                          anchor=W).grid(column=0, row=0, sticky=E)
                    Label(f, text='Q shell width: ',
                          anchor=W).grid(column=0, row=1, sticky=E)
                    Label(f, text='Vectors per shell: ',
                          anchor=W).grid(column=0, row=2, sticky=E)
                    Label(f, text='Q direction: ',
                          anchor=W).grid(column=0, row=3, sticky=E)
                    values = StringVar()
                    values.set(value[0])
                    width = StringVar()
                    width.set(value[1])
                    nvectors = StringVar()
                    nvectors.set(value[2])
                    qdir = StringVar()
                    qdir.set(value[3])
                    Entry(f, textvariable=values).grid(column=1, row=0,
                                                       sticky=W)
                    Entry(f, textvariable=width).grid(column=1, row=1,
                                                      sticky=W)
                    Entry(f, textvariable=nvectors).grid(column=1, row=2,
                                                         sticky=W)
                    Entry(f, textvariable=qdir).grid(column=1, row=3,
                                                     sticky=W)
                    form_data.append((parameter,
                                      (values, width, nvectors, qdir)))
                elif parameter == 'output_files':
                    entries = ()
                    n = 0
                    for extension, tag, id, label in self.output_files:
                        if label == '':
                            label = 'Output file '
                        else:
                            label = label + ' output file '
                        if extension == 'nc':
                            label = label + '(netCDF):'
                        else:
                            label = label + '(text):'
                        Label(f, text=label).pack(side=TOP, anchor=W)
                        entry = FilenameEntry(f,
                                              browse_pattern = '*.'+extension)
                        entry.pack(side=TOP)
                        name = string.replace(value[n], '%s',
                                              self.master.base_name)
                        entry.filename.set(name)
                        entries = entries + (entry,)
                        n = n + 1
                    form_data.append(('output_files', entries))
                else:
                    raise ValueError, "not yet implemented"

        f = Frame(window, bd=2, relief='groove')
        f.pack(side=TOP, fill=BOTH, padx=3, pady=3)
        Button1=Button(f, text='OK', underline=0,
                       command = lambda f=form_data, w=window, s=self:
                                        (w.destroy(), s.run2(f)))
        Button1.pack(padx=1, pady=1, side=LEFT)
        Button2=Button(f, text='Cancel', underline=0,
                       command = window.destroy)
        Button2.pack(padx=1, pady=1, side=RIGHT)
        window.bind('<Return>', lambda event, f=form_data, w=window, s=self:
                               (w.destroy(), s.run2(f)))
        if mac_conventions:
            window.bind('<Escape>', lambda event: window.destroy())
        else:
            window.bind('<Cancel>', lambda event: window.destroy())
        window.grab_set()
        window.initial_focus.focus_set()
        window.wait_window(window)

    def run2(self, form_data):
        lines = []
        lines.append(["from MMTK import *"])
        lines.append(["title = %s" % repr(self.name)])
        lines.append(["trajectory = %s" % str(self.master.filenames)])
        self.log_file = self.switch + '_' + self.master.base_name + '.log'
        #self.log_file = mktemp()+".log"
        lines.append(["log_file = %s" % repr(self.log_file)])
        self.master.getTime()
        lines.append(["time_info = %s" % str(self.master.TimeInfo)])
        if 'atom' in self.selections:
            lines.append(["atoms = " + str(self.master.getAtomSelection(0))])
            if self.master.atomsFromPDB is not None:
                lines.append(["atoms_pdb = " +
                              repr(self.master.atomsFromPDB[-1])])
        if 'deuter' in self.selections:
            lines.append(["deuter = " + str(self.master.getAtomSelection(1))])
        if 'group' in self.selections:
            groups, reference = self.master.getGroupSelection()
            lines.append(["groups = " + str(groups)])
            lines.append(["reference = " + str(reference)])
        error = 0
        for p, value in form_data:
            try:
                if type(value) is types.TupleType:
                    value = tuple(map(lambda v: v.get(), value))
                else:
                    value = value.get()
                self.parameters[p] = value
                if p == 'q_vector_set':
                    values, width, nvectors, qdir = value
                    width = eval(width)
                    nvectors = eval(nvectors)
                    q = []
                    for v in map(string.strip, string.split(values, ',')):
                        spec = map(float, string.split(v, ':'))
                        qstep = 1.
                        qv = spec[0]
                        qmax = qv
                        if len(spec) > 1:
                            qmax = spec[-1]
                        if len(spec) > 2:
                            qstep = spec[1]
                        while qv <= qmax:
                            q.append(qv)
                            qv = qv + qstep
                    qdir = map(string.strip, string.split(qdir, ','))
                    if len(qdir) == 1 and qdir[0] == '':
                        qdir = None
                    else:
                        if len(qdir) != 3:
                            raise ValueError
                        qdir = map(float, qdir)
                    lines.append(["q_vector_set = (%s, %f, %d, %s)"
                                  % (repr(q), width, nvectors, repr(qdir))])
                elif p == 'output_files':
                    files = {}
                    n = 0
                    for extension, tag, id, label in self.output_files:
                        if value[n]:
                            files[id] = value[n]
                        n = n + 1
                    lines.append(["output_files = " + repr(files)])
                elif p in self._string_parameters:
                    lines.append(["%s = %s" % (p, repr(value))])
                else:
                    lines.append(["%s = %s" % (p, value)])
            except:
                error = 1
        if error:
            Dialog.Dialog(self.master, title='Error',
                          text = 'One or more parameters have illegal values.', 
                          bitmap='error', default=0, strings = ('OK',))
            self.run()
            return
        for l in lines:
            if l[0][:4] == 'unit':
                l[0] = string.replace(l[0], 'nm', 'Units.nm')
                l[0] = string.replace(l[0], '\305', 'Units.Ang')
                l[0] = string.replace(l[0], 'ps', 'Units.ps')
                l[0] = string.replace(l[0], '1/cm', 'Units.invcm')
        window = Toplevel(self.master)
        window.title("Almost done...")
        window.resizable(width=NO, height=NO)
        window.initial_focus = window
        frame = Frame(window, bd=2, relief='flat')
        frame.pack(side=TOP, fill=BOTH)
        textA = Text(window,relief='flat',padx=10,pady=5,setgrid=1,width=50,
                     wrap='word',exportselection=0,height=5)
        textA.tag_config("b",foreground='black')
        textA.tag_config("r",foreground='red')
        textA.insert(END,"Your settings will stored in an input file"+\
                         " whose contents are shown below.\nYou can ",
                     ("b",))
        textA.insert(END,"Save",("r",))
        textA.insert(END," them and run the calculations later or ",("b",))
        textA.insert(END,"Run",("r",))
        textA.insert(END," the calculations immediately.",("b",))
        textA.configure(state=DISABLED)
        textA.pack(side=TOP)
        textB = ScrolledText(window, relief='ridge', padx=10, pady=5,
                             setgrid=1, width=60, height=10, wrap='none')
        for l in lines:
            textB.insert(END, l[0]+'\n')
        textB.pack(side=TOP)
        textB.configure(state=DISABLED)
        Button(window, text="Save",
               command=lambda w=window, l=lines, s=self: (w.destroy(), s.run4(l))) \
               .pack(side=LEFT)
        Button(window, text='Run',
               command=lambda w=window, l=lines, s=self: (w.destroy(), s.run3(l))) \
               .pack(side=LEFT)
        Button(window, text='Cancel',
               command=lambda w=window: w.destroy()).pack(side=RIGHT)
        window.grab_set()
        window.initial_focus.focus_set()
        window.wait_window(window)
    
    def run3(self, data):
        filename = saveText(filename=mktemp(), data=data)
        text = "The following command is about to be run in the background:"+\
               "\n\n"
        cmd  = sys.executable + ' pMoldyn.py --' + self.switch + ' --input ' + filename + \
               ' 1> '+ self.log_file + ' 2>&1 &'
        
        if askokcancel("Running...", text+cmd):
            os.system(cmd)

    def run4(self, data):
        saveText(master=self.master, data=data)


commands = [Command('Mean-Square Displacement',
                    'Dynamics', 5, 'Mean-Square Displacement',
                    ['coordinates'],
                    [('weights', 'mass'), 'projection_vector', 'units_length',
                     'time_steps'],
                    ['atom'],
                    [('plot', 'MSD', 'msd', '')],
                    'msd'),
            Command('Velocity Autocorrelation Function (from velocities)',
                    'Dynamics', 10,
                       'Velocity Autocorrelation Function|from velocities',
                    ['velocities'],
                    [('weights', 'mass'), 'projection_vector', 'units_length',
                     'time_steps'],
                    ['atom'],
                    [('plot', 'VACF', 'vacf', '')],
                    'vacf-vel'),
            Command('Velocity Autocorrelation Function (from coordinates)',
                    'Dynamics', 10,
                       'Velocity Autocorrelation Function|from coordinates',
                    ['coordinates'],
                    [('weights', 'mass'), 'projection_vector', 'units_length',
                     'differentiation', 'time_steps'],
                    ['atom'],
                    [('plot', 'VACF', 'vacf', '')],
                    'vacf-xyz'),
            Command('Density of states (from velocities)',
                    'Dynamics', 15,
                       'Density of states|from velocities',
                    ['velocities'],
                    [('weights', 'mass'), 'projection_vector', 'ft_window',
                     'units_length', 'units_frequency', 'frequency_points'],
                    ['atom'],
                    [('plot', 'DOS', 'dos', '')],
                    'dos-vel'),
            Command('Density of states (from coordinates)',
                    'Dynamics', 15,
                       'Density of states|from coordinates',
                    ['coordinates'],
                    [('weights', 'mass'), 'projection_vector', 'ft_window',
                     'units_length', 'units_frequency', 'differentiation',
                     'frequency_points'],
                    ['atom'],
                    [('plot', 'DOS', 'dos', '')],
                    'dos-xyz'),
            Command('Autoregressive Model (from velocities)',
                    'Dynamics', 20,
                       'Autoregressive Model|from velocities',
                    ['velocities'],
                    [('weights', 'mass'), 'projection_vector',
                     'units_length', 'units_frequency', 'ar_order',
                     'ar_precision', 'time_steps', 'frequency_points'],
                    ['atom'],
                    [('plot', 'AR-Memory', 'memory',
                      'Memory function output file'),
                     ('plot', 'AR-VACF', 'vacf', 'VACF output file'),
                     ('plot', 'AR-DOS', 'dos', 'DOS function output file'),
                     ('plot', 'AR-MSD', 'msd', 'MSD function output file'),
                     ('text', 'AR-Coeff', 'parameters',
                      'AR parameter output file')],
                    'ar-vel'),
            Command('Autoregressive Model (from coordinates)',
                    'Dynamics', 20,
                       'Autoregressive Model|from coordinates',
                    ['coordinates'],
                    [('weights', 'mass'), 'projection_vector',
                     'units_length', 'units_frequency', 'differentiation',
                     'ar_order', 'ar_precision', 'time_steps',
                     'frequency_points'],
                    ['atom'],
                    [('plot', 'AR-Memory', 'memory',
                      'Memory function output file'),
                     ('plot', 'AR-VACF', 'vacf', 'VACF output file'),
                     ('plot', 'AR-DOS', 'dos', 'DOS function output file'),
                     ('plot', 'AR-MSD', 'msd', 'MSD function output file'),
                     ('text', 'AR-Coeff', 'parameters',
                      'AR parameter output file')],
                    'ar-xyz'),
            Command('Angular Trajectory',
                    'Dynamics', 105, 'Angular Trajectory',
                    ['coordinates'],
                    [],
                    ['group'],
                    [('nc', 'AT', 'trajectory', '')],
                    'at'),
            Command('Rigid Body Trajectory',
                    'Dynamics', 110, 'Rigid Body Trajectory',
                    ['coordinates'],
                    [],
                    ['group'],
                    [('nc', 'RBT', 'trajectory', '')],
                    'rbt'),
            Command('Rigid Body Rotation Trajectory',
                    'Dynamics', 111, 'Rigid Body Rotation Trajectory',
                    ['coordinates'],
                    [],
                    ['group'],
                    [('nc', 'RBRT', 'trajectory', '')],
                    'rbrt'),
            Command('Digital Filter',
                    'Dynamics', 115, 'Digital Filter',
                    ['coordinates'],
                    ['filter_window'],
                    [],
                    [('nc', 'DF', 'trajectory', '')],
                    'df'),
            Command('Angular Velocity Autocorrelation Function',
                    'Dynamics', 205,
                       'Angular Velocity Autocorrelation Function',
                    ['coordinates'],
                    ['projection_vector', 'differentiation', 'time_steps'],
                    ['group'],
                    [('plot', 'AVACF', 'avacf', '')],
                    'avacf'),
            Command('Spectrum of Angular VACF',
                    'Dynamics', 210, 'Spectrum of Angular VACF',
                    ['coordinates'],
                    ['projection_vector', 'ft_window', 'differentiation',
                     'units_frequency', 'frequency_points'],
                    ['group'],
                    [('plot', 'SAVACF', 'savacf', '')],
                    'savacf'),
            Command('Reorientational Correlation Function',
                    'Dynamics', 215, 'Reorientational Correlation Function',
                    ['coordinates'],
                    ['rotation_coefficients', 'time_steps'],
                    ['group'],
                    [('plot', 'RCF', 'rcf', '')],
                    'rcf'),

            Command('Coherent Scattering Function',
                    'Scattering', 5, 'Coherent Scattering Function',
                    ['coordinates'],
                    ['q_vector_set', ('weights', 'coherent'), 'units_q',
                     'time_steps', 'frequency_points', 'ft_window'],
                    ['atom', 'deuter'],
                    [('nc', 'CSF', 'csf', ''),
                     ('nc', 'CSF_SPECT', 'fft',
                      'Output file for Dynamic Structure Factor')],
                    'csf'),
            Command('Coherent Scattering AR analysis',
                    'Scattering', 6, 'Coherent Scattering AR Analysis',
                    ['coordinates'],
                    ['q_vector_set', ('weights', 'coherent'), 'units_q',
                     'time_steps', 'frequency_points', 'ar_order',
                     'ar_precision'],
                    ['atom', 'deuter'],
                    [('nc', 'AR-CSF', 'csf',
                      'Scattering function output file'),
                     ('nc', 'AR-CSF_SPECT', 'fft',
                      'Structure factor output file'),
                     ('nc', 'AR-CSF_Memory', 'memory',
                      'Memory function output file')],
                    'arcsf'),
            Command('Incoherent Scattering Function',
                    'Scattering', 10, 'Incoherent Scattering Function',
                    ['coordinates'],
                    ['q_vector_set', ('weights', 'incoherent'), 'units_q',
                     'time_steps', 'frequency_points', 'ft_window'],
                    ['atom', 'deuter'],
                    [('nc', 'ISF', 'isf', ''),
                     ('nc', 'ISF_SPECT', 'fft',
                      'Spectrum output file')],
                    'isf'),
            Command('Incoherent Scattering Function (Gaussian approx.)',
                    'Scattering', 10,
                       'Incoherent Scattering Function (Gaussian approx.)',
                    ['coordinates'],
                    ['q_vector_set', ('weights', 'incoherent'), 'units_q',
                     'time_steps', 'frequency_points', 'ft_window'],
                    ['atom', 'deuter'],
                    [('nc', 'ISFG', 'isf', ''),
                     ('nc', 'ISFG_SPECT', 'fft', 'Spectrum output file')],
                    'isfg'),
            Command('Incoherent Scattering AR analysis',
                    'Scattering', 11, 'Incoherent Scattering AR Analysis',
                    ['coordinates'],
                    ['q_vector_set', ('weights', 'incoherent'), 'units_q',
                     'time_steps', 'frequency_points', 'ar_order',
                     'ar_precision'],
                    ['atom', 'deuter'],
                    [('nc', 'AR-ISF', 'isf',
                      'Scattering function output file'),
                     ('nc', 'AR-ISF_SPECT', 'fft',
                      'Structure factor output file'),
                     ('nc', 'AR-ISF_Memory', 'memory',
                      'Memory function output file')],
                    'arisf'),
            Command('EISF',
                    'Scattering', 105, 'EISF',
                    ['coordinates'],
                    ['q_vector_set', ('weights', 'incoherent'), 'units_q'],
                    ['atom', 'deuter'],
                    [('plot', 'EISF', 'eisf', '')],
                    'eisf'),
            ]


class xMOLDYN(Frame):

    def __init__(self,master):
 
        Frame.__init__(self,master)
        self.pack()
        self.master = master
        self.setDefault()
        self.createMenu()
        self.createChoiceWindow()
        self.setBindings()
        self.activate()

    def setBindings(self):
        self.setShortcut('o', self.openTrajectory)
        self.setShortcut('q', self.quit)

    def setShortcut(self, key, function):
        if mac_conventions:
            key = '<Command-%s>' % key
        else:
            key = '<Control-%s>' % key
        self.master.bind(key, lambda event, f=function: f())

    def setDefault(self):
 
        self.qVectorSet    = {}
        self.qVecSpec = None
 
        self.normEntry = {}
        self.freqEntry = None
        self.orderEntry = {}
        self.jmn  = None
        self.symbols = None
        self.lu_last = {}
        self.traj = None
        self.result = {}

        self.weightType = {}
        self.Weights = 0

        self.frequencyUnits = {}
        self.lengthUnits = {}
        self.settings = {'lengthUnits':0,'frequencyUnits':0}
        self.setGroup = None
        self.setAtom = None
        self.setDeuter = None
        self.setReference = None
        self.setFilter = (0.0,None)
        self.verbose = IntVar()
        self.verbose.set(0)
        self.groupNumber = StringVar()
        self.groupNumber.set('One')
        self.diff_scheme = {}
        self.atomsFromPDB = None
        self.Data = {}
        self.filenameNetCDFResults = None
 
        self.t0    = 0
        self.t00   = 1
        self.tstep = 1
        self.TimeStep = 0.001

        self.available_data = []

        self.AnalScattering              = 0
        self.AnalDynamics                = 0
        self.ViewVariablesList           = []
        self.proceed                     = 0
 
        self.command = 'pMoldyn '


###############################
#                             #
#    Set up the menus         #
#                             #
###############################

    def createMenu(self):

        menu_bar = Menu(self)
        self.createFileMenu(menu_bar)
        self.createCommandMenus(menu_bar)
        self.createViewMenu(menu_bar)
        self.createHelpMenu(menu_bar)
        self.master.config(menu = menu_bar)

    def createFileMenu(self, menu_bar):
 
        if mac_conventions:
            command_key = 'Command-'
        else:
            command_key = 'Ctrl-'
        menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=menu)
        menu.add_command(label='Open trajectory...',
                         command=self.openTrajectory,
                         accelerator=command_key+'O')
        menu.add_command(label='Open trajectory set...',
                         command=self.openTrajectorySet)
        menu.add_separator()
        menu.add_command(label='Open scattering function...',
                         command=self.openResultsScattering)
        menu.add_command(label='Write PDB file...', command=self.writePDB)
        menu.add_command(label='Show running calculations ',
                         command=self.showProgress)
        menu.add_separator()
        menu.add_command(label='Quit', command=self.quit,
                         accelerator=command_key+'Q')
        self.file_menu = menu

    def createCommandMenus(self, menu_bar):
        entries = {}
        menu_order = {}
        for command in commands:
            entries[command.menu_title] = entries.get(command.menu_title, {})
            menu_order[command.menu_entry[0]] = command.menu_position
            dict = entries[command.menu_title]
            for e in command.menu_entry:
                dict[e] = dict.get(e, {})
                last_dict = dict
                dict = dict[e]
            last_dict[e] = command
        menu_titles = entries.keys()
        menu_titles.sort()
        self.command_menus = {}
        for title in menu_titles:
            menu = Menu(menu_bar, tearoff=0)
            menu_bar.add_cascade(label=title, menu=menu)
            self.command_menus[title] = menu
            main_entries = entries[title].keys()
            main_entries = map(lambda e, m=menu_order: (e, m[e]), main_entries)
            main_entries.sort(lambda a, b: cmp(a[1], b[1]))
            main_entries = map(lambda a: a[0], main_entries)
            entry_number = -1
            section_number = 0
            for entry in main_entries:
                entry_number = entry_number + 1
                section = menu_order[entry] / 100
                if section > section_number:
                    section_number = section
                    menu.add_separator()
                    entry_number = entry_number + 1
                action = entries[title][entry]
                if type(action) is types.DictType:
                    sub_menu = Menu(menu_bar, tearoff=0)
                    menu.add_cascade(label=entry+'...', menu=sub_menu)
                    sub_entries = action.keys()
                    sub_entries.sort()
                    sub_entry_number = -1
                    for sub_entry in sub_entries:
                        sub_entry_number = sub_entry_number + 1
                        command = action[sub_entry]
                        sub_menu.add_command(label=sub_entry,
                                             command=getattr(command, 'run'))
                        command.menu = sub_menu
                        command.menu_entry_number = sub_entry_number
                        command.master = self
                else:
                    menu.add_command(label=entry+'...',
                                     command=getattr(action, 'run'))
                    action.menu = menu
                    action.menu_entry_number = entry_number
                    action.master = self

    def createViewMenu(self, menu_bar):
 
        menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="View", menu=menu)
        menu.add_command(label='Trajectory Information',command=self.info)
        menu2=Menu(menu_bar, tearoff=0)
        cascade=menu.add_cascade(label='View Variables',menu=menu2)
        menu.add_command(label='Animation...',command=self.animation)
        menu.add_separator()
        menu.add_command(label='Display',command=self.PolyDisplay)
        menu.add_command(label='Display 3D',command=self.Display3D)
        self.ViewMenu = menu 
        self.ViewVariablesMenu = menu2       

    def createHelpMenu(self, menu_bar):
 
        menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=menu)
        menu.add_command(label='About nMOLDYN',command=self.aboutInfo)
        self.help_menu = menu

 
###############################
#                             #
#   Set up the main window    #
#                             #
###############################

    def createChoiceWindow(self):
 
        frame = Frame(self, bd=2)
        frame.pack(side=TOP, fill=Y)
        f11 = Frame(frame,bd=2,relief='groove')
        f11.pack(side=TOP, padx=7, pady=7)

        f13 = Frame(f11)
        f13.pack(side=TOP, padx=7, pady=7)
        l1 = Label(f13,text='First step',anchor=W)
        l1.grid(column=0,row=0,sticky='news',pady=7)
        self.myFont = Font(font=l1["font"]).copy()
        self.e41 = IntEntry(f13,'', self.t0)
        self.e41.grid(column=1,row=0,pady=7)
        self.e41.bind('<Return>',lambda event,f=frame,s=self:
                                        s._changeTimeInfo(f))
        self.t41 = Label(f13, text='(%f ps)' % (self.t0*self.TimeStep),
                         anchor=W)
        self.t41.grid(column=2, row=0, pady=7)

        Label(f13,text='Last step',anchor=W).grid(column=0,row=1,
                                                  sticky='news',pady=7)
        self.e42 = IntEntry(f13,'',self.t00)
        self.e42.grid(column=1,row=1,pady=7)
        self.e42.bind('<Return>',lambda event,f=frame,s=self:
                                        s._changeTimeInfo(f))
        self.t42 = Label(f13, text='(%f ps)' % (self.t00*self.TimeStep),
                         anchor=W)
        self.t42.grid(column=2, row=1, pady=7)
 
        Label(f13,text='Skip',anchor=W).grid(column=0,row=2,
                                             sticky='news',pady=7)
        self.e43 = IntEntry(f13,'',self.tstep)
        self.e43.grid(column=1,row=2,pady=7)
        self.e43.bind('<Return>',lambda event,f=frame,s=self:
                                        s._changeTimeInfo(f))
        self.t43 = Label(f13, text='(%f ps)' % (self.tstep*self.TimeStep),
                         anchor=W)
        self.t43.grid(column=2, row=2, pady=7)
 
        fxx = Frame(f11,bd=2,relief='groove')
        fxx.pack(side=TOP, padx=7, pady=7)
        fxxx = Frame(fxx)
        fxxx.pack(side=TOP)
        self.AtomButton=Button(fxxx,text='Atom Selection',
                               command=self.AtomSelection)
        self.AtomButton.grid(row=0,column=0,sticky='news')
#          self.GroupButton = Menubutton(fxxx,text='Group selection',
#                                        bd=2,relief='raised')
#          menu_group = Menu(self.GroupButton,tearoff=0)
#          menu_group.add_command(label='Selection ONE',
#                                 command=(lambda self=self,var='One': \
#                                          self.GroupSelection(var)))
#          menu_group.add_command(label='Selection TWO',
#                                 command=(lambda self=self,var='Two': \
#                                          self.GroupSelection(var)))
#          self.GroupButton.config(menu=menu_group)
        self.GroupButton=Button(fxxx,text='Group Selection',
                                command=(lambda self=self,var='One': \
                                         self.GroupSelection(var)))
        self.GroupButton.grid(row=0,column=1,sticky='news',padx=1,pady=1)
 
        self.status = StatusBar(self)
        self.status.pack(side=BOTTOM, anchor=W, fill=BOTH)

    def _updateTimeInfo(self):
        self.t41.config(text='(%.3f ps)' % (self.t0*self.TimeStep))
        self.t42.config(text='(%.3f ps)' % (self.t00*self.TimeStep))
        self.t43.config(text='(%.3f ps)' % (self.tstep*self.TimeStep))

    def _changeTimeInfo(self, frame):
        self.getTime()
        self._updateTimeInfo()
        frame.focus_set()

#####################
#                   #
#   Disable/Unable  #
#                   #
#####################

    def activate(self):
        for command in commands:
            if command.isActive(self.available_data):
                state = NORMAL
            else:
                state = DISABLED
            command.menu.entryconfig(command.menu_entry_number, state = state)

        if self.AnalScattering == 1:
            self.ViewMenu.entryconfig(4,state=NORMAL)
            self.ViewMenu.entryconfig(5,state=NORMAL)
        else:
            self.ViewMenu.entryconfig(4,state=DISABLED)
            self.ViewMenu.entryconfig(5,state=DISABLED)

    def loadTrajectory(self, filename):

        self.filenames=[filename]
        self.base_name = path.splitext(path.split(filename)[1])[0]
        self.status.set("Loading trajectory...")
        try:
            self.traj = Trajectory(None,filename,'r')
            if 'quaternion' in self.traj.variables():
                self.traj = qTrajectory(None,filename,'r')
            self.types = getTypes(self.traj.universe)
            self.available_data = []
            for v in self.traj.variables():
                if v == 'configuration':
                    self.available_data.append('coordinates')
                if v == 'velocities':
                    self.available_data.append('velocities')
                if v == 'quaternion':
                    self.available_data.append('quaternions')
            self.atomButtonCollection  = {}
            self.groupButtonCollection = {'One':{},'Two':{}}
            self.menuProtein           = {'One':{},'Two':{}}
            self.backbone = {}
            self.methyl = {}
            self.atomsFromPDB = None
            try:
                self.TimeStep = self.traj.time[1]-self.traj.time[0]
                self.trajInfo = trajectoryInfo(filename)
                self.t00 = len(self.traj)-1
                self.e42.set(self.t00)
                self._updateTimeInfo()
            except:
                Dialog.Dialog(self, title='Error',
                       text='Trajectory does not include time information',
                       bitmap='error', default=0, strings = ('Cancel',))
                self.available_quantities = []
            self.activate()
            self.NewViewVariablesMenu()
            for command in commands:
                command.setDefaults()
        except:
            Dialog.Dialog(self, title='Read Error',
                          text='This is most probably not a trajectory file', 
                          bitmap='error', default=0,strings = ('Cancel',))
            raise
        self.status.clear()

##############################
#                            #
#   Actions from FILE Menu   #
#                            #
##############################

    def openTrajectory(self):

        #fd = FileDialog.LoadFileDialog(self)
        #filename = fd.go(key='LoadTrajectory', pattern='*.nc')
        open = tkFileDialog.Open(filetypes=[("netCDF files", "*.nc"),
                                            ("All files", "*")],
                                 title = "Open trajectory")
        filename = open.show()
        if not filename: return
        self.loadTrajectory(filename)

    def openTrajectorySet(self):

        #fd = FileDialog.LoadFileDialog(self)
        #filename = fd.go(key='LoadTrajectorySet', pattern='*.ncs')
        open = tkFileDialog.Open(filetypes=[("Trajectory sets", "*.ncs"),
                                            ("All files", "*")],
                                 title = "Open trajectory set")
        filename = open.show()
        if not filename: return
        self.status.set("Loading trajectory...")
        self.base_name = path.splitext(path.split(filename)[1])[0]
        file = TextFile(filename, 'r')
        filenames=[]
        for line in file:
            if len(line)>0:
                filenames.append(str(line)[:-1])
        self.filenames=filenames
        try:
            self.traj = TrajectorySet(None,filenames)
            if 'quaternion' in self.traj.variables():
                self.traj = qTrajectory(None,filename,'r')
            self.types = getTypes(self.traj.universe)
            self.available_quantities = []
            for v in self.traj.variables():
                if v == 'configuration':
                    self.available_quantities.append('coordinates')
                if v == 'velocities':
                    self.available_quantities.append('velocities')
                if v == 'quaternion':
                    self.available_quantities.append('quaternions')
            self.atomButtonCollection  = {}
            self.groupButtonCollection = {'One':{},'Two':{}}
            self.menuProtein           = {'One':{},'Two':{}}
            self.backbone = {}
            self.methyl = {}
            self.atomsFromPDB = None 
            try:
                self.TimeStep = self.traj.time[1]-self.traj.time[0]
            except:
                Dialog.Dialog(self, title='Error',
                       text='Trajectory does not include time information',
                       bitmap='error', default=0, strings = ('Cancel',))
                self.available_quantities = []
            try:
                self.trajInfo = trajectoryInfo(filenames[0])
                self.t00 = len(self.traj)-1
                self.e42.set(self.t00)
                self._updateTimeInfo()
            except:
                pass
        except :
            Dialog.Dialog(self, title='Read Error',
                          text='This is most probably not a trajectory set file', 
                          bitmap='error', default=0,strings = ('Cancel',))
        self.activate()
        self.NewViewVariablesMenu()
        self.status.clear()

    def NewViewVariablesMenu(self):
        if len(self.ViewVariablesList) >0:
            self.ViewVariablesMenu.delete(1,len(self.ViewVariablesList))
            self.ViewVariablesList=[]
        self.categories = categorizeVariables(self.traj.variables())
        self.time = self.traj.time
        step_number = self.traj.step
        jumps = less(step_number[1:]-step_number[:-1], 0)
        self.restarts = repeat(arange(len(jumps)), jumps)+1
        self.restarts = list(self.restarts)
        categories = self.categories.keys()
        categories = filter(lambda c, cl=categories: c in cl,
                        ['energy', 'thermodynamic', 'auxiliary'])
        for category in categories:
            self.ViewVariablesMenu.add_command(label=string.capitalize(category),
                         command=lambda s=self, c=category: s.ViewData(c))
            self.ViewVariablesList.append(category)
        self.ViewMenu.entryconfig(2,state=NORMAL)

        
    def openResultsScattering(self):
        """ open results stored in the NetCDF format
        (e.g. from Scattering menu) """
        
        #fd = FileDialog.LoadFileDialog(self)
        #filename = fd.go(key='LoadScattering', pattern='*.nc')
        open = tkFileDialog.Open(filetypes=[("netCDF files", "*.nc"),
                                            ("All files", "*")],
                                 title = "Open scattering function")
        filename = open.show()

        if not filename: return
        self.resultsScattering = NetCDFFile(filename, 'r')
        self.resultsVariable = None
        for name in ['sf', 'dsf', 'sfmemory']:
            try:
                self.resultsVariable = self.resultsScattering.variables[name]
                break
            except KeyError:
                pass
        if self.resultsVariable is None:
            Dialog.Dialog(self, title='Read Error',
                          text='This is not a result file', 
                          bitmap='error', default=0,
                          strings = ('Cancel',))
        else:
            self.AnalScattering = 1
            self.filenameNetCDFResults = filename
            try: self.title2 = self.resultsScattering.title
            except: self.title2 = None
        self.activate()

#
#    def openResultsDynamics(self):
#        """ open results stored in the ASCII format
#        (e.g. from Dynamics menu) """
#        
#        fd = FileDialog.LoadFileDialog(self)
#        filename = fd.go(key='LoadConformation', pattern='*.dat')
#        self.filenameASCIIResults = filename
#        self.AnalDynamics = 0
#        if not filename: return
#        try:
#            self.resultsDynamics = getData(filename)
#            if self.resultsDynamics: self.AnalDynamics = 1
#        except:
#            Dialog.Dialog(self, title='Reading Error',
#                          text='This is not a result file', 
#                          bitmap='error', default=0,
#                          strings = ('Cancel',))
#        self.activate()
#

    def writePDB(self):
        """ write out a PDB file which can be later used as a template
        to select some atoms, groups or atoms or to define a reference
        configuration for choosen groups of atoms """
        
        #fd       = FileDialog.SaveFileDialog(self)
        #filename = fd.go(key='SavePDB', pattern='*.pdb')
        open = tkFileDialog.SaveAs(filetypes=[("PDB files", "*.pdb"),
                                              ("All files", "*")],
                                 title = "Write PDB file")
        filename = open.show()
        self.status.set("Writing PDB file...")
        self.getTime()
        if filename:
            PDBOutputFile(filename).write(self.traj.universe,
                                          self.traj.configuration[self.t0])
        self.status.clear()


    def showProgress(self):
        """ Provide info about currently running calculations.
        The name of a job (eg. the module name), a job ID,
        a job owner, the date of starting
        and a progress in percents are shown """

        try:
            import win32api
            owner = win32api.GetUserName()
        except ImportError:
            from pwd import getpwuid
            owner = getpwuid(getuid())[0]
        tempdir = gettempdir()
        list = listdir(tempdir)
        list2 = []
        for file in list:
            try:
                a = split(file,'.')
                if a[-1] == 'moldyn' and a[-4] == owner:
                    jobID = int(a[-3])
                    try:
                        kill(jobID, 0)
                        list2.append(file)
                    except:
                        unlink(file)
            except:
                pass
        if len(list2) == 0:
            Dialog.Dialog(self, title='Information', \
                text='No currently running calculations', \
                bitmap='error', default=0, \
                strings = ('Cancel',))
        else:
            self.ind = Toplevel(self)
            self.ind.title('Progress indicator')
            f0 = Frame(self.ind)
            f0.pack(side=TOP, padx=3, pady=3)
            f1 = Frame(f0,bd=2,relief='groove') 
            f1.pack(side=LEFT, padx=3, pady=3)
            Label(f1,text='Module' ).pack(side=TOP)
            f4 = Frame(f0,bd=2,relief='groove')
            f4.pack(side=LEFT, padx=3, pady=3)
            Label(f4,text='Job ID').pack(side=TOP)
            f5 = Frame(f0,bd=2,relief='groove')
            f5.pack(side=LEFT, padx=3, pady=3)
            Label(f5,text='Job Owner').pack(side=TOP) 
            f2 = Frame(f0,bd=2,relief='groove')
            f2.pack(side=LEFT, padx=3, pady=3)
            Label(f2,text='Started', height=1,width=25).pack(side=TOP)
            f3 = Frame(f0,bd=2,relief='groove')
            f3.pack(side=LEFT, padx=3, pady=3)
            Label(f3,text='Progress in %').pack(side=TOP)
            for file in list2:
                record = split(file,'.')
                mod = record[-2]
                jobID = int(record[-3])
                jobOwner = record[-4]
                temp = tempdir
                filename = path.join(temp, file)
                f = TextFile(filename,'r')
                l=[]
                for line in f: l.append(line[:-1])
                progress = IntVar()
                if len(l) > 1:
                    try:
                        t=l[0]
                        prog=l[len(l)-1]
                        progress.set(int(prog))
                        Label(f1,text=mod ).pack(side=TOP)
                        Label(f4,text=jobID).pack(side=TOP)
                        Label(f5,text=jobOwner).pack(side=TOP)
                        Label(f2,text=t).pack(side=TOP)
                        Label(f3,textvariable=progress).pack(side=TOP)
                    except: pass


###############################
#                             #
#  Action from the View Menu  #
#                             #
###############################

    def animation(self):

        dialog = Toplevel(self)
        dialog.title('Trajectory animation')
        f1 = Frame(dialog)
        f1.pack(side=TOP,fill=BOTH)
        Label(f1, text='Trajectory animation').pack(side=TOP,anchor=W,pady=7)

        f11 = Frame(f1,bd=2,relief='groove')
        f11.pack(side=TOP,fill=BOTH,padx=5,pady=5)
        Label(f11, text='First step').grid(row=0,column=0,sticky='w')
        Label(f11, text='Last step').grid(row=1,column=0,sticky='w')
        Label(f11, text='Skip').grid(row=2,column=0,sticky='w')
        n = len(self.traj)-1
        first = IntEntry(f11, '', 0, 0, n)
        first.grid(row=0,column=1,sticky='w')
        last = IntEntry(f11, '', n, 0, n)
        last.grid(row=1,column=1,sticky='w')
        skip = IntEntry(f11, '', max(1, n/100), 1, n)
        skip.grid(row=2,column=1,sticky='w')

        f12 = Frame(f1,bd=2,relief='groove')
        f12.pack(side=TOP,fill=X)
        Button1=Button(f12,text='OK',
                       command=lambda self=self, d=dialog,
                                      f=first, l=last, s=skip:
                               self.do_animation(d, f, l, s),
                       underline=0)
        Button1.pack(padx=1,pady=1,side=LEFT)
        Button2=Button(f12,text='Cancel',command=dialog.destroy,underline=0)
        Button2.pack(padx=1,pady=1,side=RIGHT)

        dialog.grab_set()
        dialog.wait_window(dialog)

    def do_animation(self, dialog, first, last, skip):
        from MMTK.Visualization import viewTrajectory
        dialog.destroy()
        viewTrajectory(self.traj, first.get(), last.get(), skip.get())

    def Display3D(self):

        self.status.set("Processing data ...")
        self.displ = Toplevel(self)
        if self.title2 is None: pass
        else: self.displ.title(self.title2)

        qlengths = self.resultsScattering.variables['q'][:]
        if self.resultsScattering.variables.has_key('time'):
            timefreq = self.resultsScattering.variables['time'][:]
        else:
            timefreq = self.resultsScattering.variables['frequency'][:]
        data = self.resultsVariable

        lowq = data[0, :]
        tfmax = sum(logical_and.accumulate(greater(lowq,
                                                   0.02*maximum.reduce(lowq))))
        tfmax = min(tfmax+tfmax/5, len(lowq)-1)
        
        qskip = max(1, len(qlengths)/50)
        tskip = max(1, tfmax/50)
        xscale = 1./qlengths[-1]
        yscale = 1./timefreq[tfmax]
        zscale = 1./maximum.reduce(data[:, 0])
        qlengths = qlengths*xscale
        timefreq = timefreq*yscale
        objects = []
        for iq in range(0, len(qlengths), qskip):
            x = qlengths[iq]
            objects.append(PolyLine3D([(x, timefreq[0], 0.),
                                       (x, timefreq[tfmax], 0.)],
                                      color='blue'))
            y = timefreq[0:tfmax:tskip]
            z = zscale*data[iq, 0:tfmax:tskip]
            x = x*ones(y.shape, Float)
            objects.append(PolyLine3D(transpose([x, y, z]),
                                      color='green'))
        for it in range(0, tfmax, tskip):
            y = timefreq[it]
            objects.append(PolyLine3D([(qlengths[0], y, 0.),
                                       (qlengths[-1], y, 0.)],
                                      color='blue'))
            x = qlengths[0::qskip]
            z = zscale*data[0::qskip, it]
            y = y*ones(x.shape, Float)
            objects.append(PolyLine3D(transpose([x, y, z]),
                                      color='green'))
        objects.append(PolyLine3D([(0., 0., 0.), (1.02, 0., 0.)],
                                  color='red', width=2))
        objects.append(PolyLine3D([(0., 0., 0.), (0., 1.02, 0.)],
                                  color='red', width=2))
        objects.append(PolyLine3D([(0., 0., 0.), (0., 0., 1.02)],
                                  color='red', width=2))
        f1 = Frame(self.displ)
        f1.pack(side=TOP, fill=BOTH, expand=YES)
        canvas = VisualizationCanvas(f1, "150m", "150m")
        canvas.pack(side=TOP, fill=BOTH, expand=YES)
        axis = Vector(1., 1., 0.5).normal()
        v1 = Vector(0., 0., 1.).cross(axis).normal()
        v2 = axis.cross(v1).normal()
        canvas.setViewpoint(axis.array, transpose([-v1.array, v2.array]))
        canvas.draw(VisualizationGraphics(objects))

        self.status.clear()

    def PolyDisplay(self):

        self.status.set("Processing data ...")
        now1 = self.resultsVariable[:,0]
        now2 = self.resultsVariable[0,:]
        qlengths = self.resultsScattering.variables['q'][:]
        self.QStep = qlengths[1]-qlengths[0]
        
        if self.resultsScattering.variables.has_key('time'):
            data = self.resultsScattering.variables['time'][:]
            self.Data['timefreq'] = data
            self.Data['label'] = 'Time = '
            self.Data['name'] = 'time'
            unit = 'ps'
        elif self.resultsScattering.variables.has_key('frequency'):
            data = self.resultsScattering.variables['frequency'][:]
            self.Data['timefreq'] = data
            self.Data['label'] = 'Frequency = '
            self.Data['name'] = 'frequency'
            unit = '1/ps'
        self.Data['step'] = data[1]-data[0]

        cut = len(now2)-sum(logical_and.accumulate(
                 less(now2[::-1], 0.02*maximum.reduce(now2))))
        cut = min(cut+cut/5, len(now2))
        cut = len(now2)
        if cut > 100: skip = cut/100
        else: skip = 1
        self.Plot1 = transpose(array([qlengths,now1]))
        self.Plot2 = transpose(array([data[:cut:skip],now2[:cut:skip]]))
         
        self.displ = Toplevel(self)
        self.displ.resizable(width=YES,height=YES)
        if self.title2 is None: pass
        else: self.displ.title(self.title2)
 
        f1 = Frame(self.displ)
        f1.pack(side=TOP,fill=BOTH,expand=YES)
        
        f11 = Frame(f1,relief=SUNKEN,border=2)
        f11.pack(side=TOP, fill=BOTH, expand=YES)
        self.c1 = PlotCanvas(f11,width=300,height=200,zoom=1)
        self.c1.pack(side=TOP, fill=BOTH, expand=YES)
        Label(f11, text='q [1/nm]', bg='white').pack(side=TOP, fill=X,
                                                     expand=YES)
        l1 = PolyLine(self.Plot1, color='red')
        self.c1.draw(l1, 'automatic', 'automatic')
        f12 = Frame(f1)
        f12.pack(side=TOP,anchor=W)
        button1 = Button(f12,text='Save plot',command=(lambda self=self: \
                         saveText(master=self,data=self.Plot1)))
        button1.pack(side=LEFT)
        l1 = Label(f12,text=self.Data['label'])
        l1.pack(side=LEFT)
        self.e71 = FloatEntry(f12,'', self.Data['timefreq'][0])
        self.e71.pack(side=LEFT)
        self.e71.bind('<Return>',self.res1)
        l11 = Label(f12,text=unit)
        l11.pack(side=RIGHT)
 
        f3 = Frame(self.displ)
        f3.pack(side=TOP,fill=BOTH,expand=YES)
        f31 = Frame(f3,relief=SUNKEN,border=2)
        f31.pack(side=TOP, fill=BOTH, expand=YES)
        self.c2 = PlotCanvas(f31,width=300,height=200,zoom=1)
        self.c2.pack(side=TOP, fill=BOTH, expand=YES)
        if self.resultsScattering.variables.has_key('time'):
            Label(f31, text='Time [ps]', bg='white').pack(side=TOP, fill=X,
                                                          expand=YES)
        else:
            Label(f31, text='Frequency [1/ps]', bg='white').pack(side=TOP,
                                                                 fill=X,
                                                                 expand=YES)
        l3 = PolyLine(self.Plot2, color='red') #
        self.c2.draw(l3, 'automatic', 'automatic')
 
        f32 = Frame(f3)
        f32.pack(side=TOP,anchor=W)
        button2 = Button(f32,text='Save plot',command=(lambda self=self: \
                         saveText(master=self,data=self.Plot2)))
        button2.pack(side=LEFT)
        l1 = Label(f32,text='Length =')
        l1.pack(side=LEFT)
        self.e72 = FloatEntry(f32,'',self.resultsScattering.variables['q'][0])
        self.e72.pack(side=LEFT,expand=YES)
        self.e72.bind('<Return>',self.res2)
        l12 = Label(f32,text='1/nm')
        l12.pack(side=RIGHT)
        self.status.clear()


    def res2(self,event):

        timefreq = self.Data['timefreq']
        qlengths = (self.resultsScattering.variables['q'][:])
        a = min(sum(less(qlengths, self.e72.get())), len(qlengths)-1)
        self.e72.set(qlengths[a])
        data = self.resultsScattering.variables[self.Data['name']][:]
        now2 = self.resultsVariable[a, :]
        cut = len(now2)-sum(logical_and.accumulate(
                 less(now2[::-1], 0.02*maximum.reduce(now2))))
        cut = min(cut+cut/5, len(now2))
        if cut > 100: skip = cut/100
        else: skip = 1
        self.Plot2 = transpose(array([timefreq[:cut:skip],now2[:cut:skip]]))
        self.c2.clear()
        l2 = PolyLine(self.Plot2, color='red')
        self.c2.draw(l2, 'automatic', 'automatic')
          
    def res1(self,event):

        timefreq = self.Data['timefreq']
        qlengths = (self.resultsScattering.variables['q'][:])
        a = min(sum(less(timefreq, self.e71.get())), len(timefreq)-1)
        self.e71.set(timefreq[a])
        now1 = self.resultsVariable[:, a]
        self.Plot1 = transpose(array([qlengths, now1]))
        self.c1.clear()
        l1 = PolyLine(self.Plot1, color='red')
        self.c1.draw(l1, 'automatic', 'automatic')
 
    def FourierTransform(self):
        """ fft of Scattering Functions stored in the NetCDF file """

        self.option_go('SFFT')
        if self.proceed == 0: return
        try:
            self.filename2 = {'fft': self.result['SFFT'][1]}
            self.TimeInfo = None; self.qVectorSet = None
            self.title1 = 'Dynamic Structure Factor'
        except:
            self.gj()
            self.option_go('SFFT')
            return
        inputFile = self.inputFilePrepare('SFFT')
        inputFile = self.inputFileFilter(['resu','outp','units_f',
                                          'titl','log_'],inputFile)
        fd = fancyDialog(self,"Almost done...",inputFile,
                         ' --fft --input ')
        if fd.cmd:
            os.system(fd.cmd)
            

    def Fitting(self):
        """
        Perform a linear regression for (x,y) read
        """
        x = self.resultsDynamics[:,0]
        y = self.resultsDynamics[:,1]
        xy = transpose(array([x,y]))
        xy = take(xy,range(0,len(xy),int(len(xy)/100)))
        if len(x) != len(y):
            Dialog.Dialog(self, title='Error', 
                text='The number of points in the input vectors\
                      does not match',
                bitmap='error', default=0, 
                strings = ('Cancel',))
        else:
            self.fit = linearRegression(x,y)
            self.fit = self.fit + [self.fit[0]+self.fit[1]*x[0],\
                                   self.fit[0]+self.fit[1]*x[-1]]
            self.fit_win = Toplevel(self)
            self.fit_win.title('Results of fitting')
            f1 = Frame(self.fit_win)
            f1.pack(side=TOP,fill=BOTH)
            f11 = Frame(f1)
            f11.pack(side=TOP)
            f111 = Frame(f11,bd=2,relief='groove')
            f111.pack(side=LEFT,padx=3,pady=3)
            Label(f111,text='Data read ('+str(len(x))+' points)')\
                       .grid(row=0,column=0,sticky=W)
            f112 = Frame(f11)
            f112.pack(side=LEFT,padx=3,pady=3)
            Button(f112,text='Save plot',command=self.saveFit)\
                        .grid(row=0,column=1,sticky=E)
          #
          #  xmgr is slightly better in this aspect
          #
          #  f12 = Frame(f1)
          #  f12.pack(side=TOP,fill=BOTH,padx=3,pady=3)
          #  self.pc1 = PlotCanvas(f12,width=300,height=200,
          #                        relief=SUNKEN,border=2)
          #  self.pc1.pack(side=TOP,fill=BOTH,expand=YES)
          #  l1 = PolyMarker(xy,color='red',
          #                     marker='dot',fillcolor='red')
          #  self.pc1.draw(l1,'automatic','automatic')
          #  l2 = PolyLine([[x[0],self.fit[-2]],[x[-1],self.fit[-1]]],
          #                color='green')
          #  self.pc1.draw(l2,'automatic','automatic')
          #
            f13 = Frame(f1,bd=2,relief='groove')
            f13.pack(side=TOP,padx=3,pady=3)
            Label(f13,text='y = a + bx').grid(row=0,columnspan=2,sticky=W+E)
            Label(f13,text='a').grid(row=1,column=0,sticky=W)
            Label(f13,text=str(self.fit[0])).grid(row=1,column=1,sticky=E)
            Label(f13,text='b').grid(row=2,column=0,sticky=W)
            Label(f13,text=str(self.fit[1])).grid(row=2,column=1,sticky=E)
            Label(f13,text='sig a').grid(row=3,column=0,sticky=W)
            Label(f13,text=str(self.fit[2])).grid(row=3,column=1,sticky=E)
            Label(f13,text='sig b').grid(row=4,column=0,sticky=W)
            Label(f13,text=str(self.fit[3])).grid(row=4,column=1,sticky=E)
            Label(f13,text='r coef.').grid(row=5,column=0,sticky=W)
            Label(f13,text=str(self.fit[4])).grid(row=5,column=1,sticky=E)
            Label(f13,text='chi2').grid(row=6,column=0,sticky=W)
            Label(f13,text=str(self.fit[5])).grid(row=6,column=1,sticky=E)
            Label(f13,text='1st fitted point (x,y)').grid(row=7,
                      columnspan=2,sticky=W+E)
            Label(f13,text=str(x[0])).grid(row=8,column=0,
                      sticky=W+E)
            Label(f13,text=str(self.fit[-2])).grid(row=8,column=1,
                      sticky=W+E)
            Label(f13,text='last fitted point (x,y)').grid(row=9,
                      columnspan=2,sticky=W+E)
            Label(f13,text=str(x[-1])).grid(row=10,column=0,
                      sticky=W+E)
            Label(f13,text=str(self.fit[-1])).grid(row=10,column=1,
                      sticky=W+E)


    def saveFit(self):
        a = self.resultsDynamics.tolist()
        a[0].append(self.fit[-2])
        a[-1].append(self.fit[-1])
        self.textData = a
        self.saveText()


###################################
#                                 #
#   Action from the Help Menu  #
#                                 #
###################################
 
    def aboutInfo(self):
        """ info about program authors and history """
        about = Toplevel(self)
        about.title('About nMoldyn')
        f0 = Frame(about,bd=2,relief='groove')
        f0.pack(side=TOP,padx=3,pady=3,fill=BOTH)
        self.logo = PhotoImage(data=nMoldyn.docs.logo)
        Label(f0,image=self.logo).grid(column=0,row=0)
        Label(f0,text=nMoldyn.docs.aboutAuthors,justify=LEFT).grid(column=1,
                                                                   row=0)
#        tb = Text(f0,relief='ridge',padx=10,pady=5,setgrid=1,width=60,
#                  height=10,wrap='none',exportselection=0)
#        tb.insert(END,nMoldyn.docs.aboutHistory)
#        tb.configure(state=DISABLED)
#        tb.grid(column=0,row=1,columnspan=2)
#        vbar = Scrollbar(f0,name='vbar')
#        vbar.grid(column=2,row=1,columnspan=2,sticky='ns')
#        tb.configure(yscrollcommand=vbar.set)
#        vbar.configure(command=tb.yview)

        f2 = Frame(about,bd=2,relief='groove')
        f2.pack(side=TOP,fill=X,padx=3,pady=3)
        Button1=Button(f2,text='Close',
                       command=about.destroy,
                       underline=0)
        Button1.pack(padx=1,pady=1,side=RIGHT)


        about.resizable(width=NO,height=NO)
        about.initial_focus = about
        about.grab_set()
        about.initial_focus.focus_set()
        about.wait_window(about)


#########################################
#                                       #
#    Functions used in Menus action     #
#                                       #
#########################################


    def getTime(self):
 
        self.t0    = self.e41.get()
        self.t00   = self.e42.get()  # <t0,t00)
        self.tstep = self.e43.get()
        self.TimeInfo = (self.t0,self.t00,self.tstep)

    def getAtomSelection(self,which):

        var_list = {}
        for k in self.types.keys():
            try:
                gj = self.atomButtonCollection[k]
                var_list[k] = []
                for ia in gj.keys():
                    v1 = gj[ia][0][which].get()
                    if v1 == 1: var_list[k].append(ia)
            except:                                         # TypeError:
                if   which == 0: var_list[k] = ['All']      # default ATOMS
                elif which == 1: var_list[k] = ['None']     # default DEUTER
        for k in var_list.keys(): # test
            if var_list[k][0] == 'None': del var_list[k]
            elif var_list[k][0] == 'All': var_list[k] = ['*']
        if len(var_list) == 0: var_list = None
        return var_list


    def getGroupSelection(self,number=['One']):

        if len(number) == 2:
            group = [ {}, {} ]
            ref   = [ {}, {} ]
        else:
            group = [ {} ]
            ref   = [ {} ]
            del number[1:] # only the first group
        
        for grn in range(len(number)):
            
            data = self.groupButtonCollection[number[grn]]
            if len(data.keys()) == 0:
                Dialog.Dialog(self, title='Error',
                       text='You must define group <'+number[grn]+'>', 
                       bitmap='error', default=0,
                       strings = ('Proceed',))
                self.GroupSelection(number[grn])
                data = self.groupButtonCollection[number[grn]]

            for title in data.keys():
                master = split(title)
                if len(master) == 2: # Proteins
                    item = master[0] + ' ' + self.menuProtein\
                           [number[grn]][master[0]].get()
                    if title == item:
                        ref[grn][item] = {}
                        group[grn][item] = []
                        for set in data[item].keys():
                            if data[item][set][0][0].get():
                                ref[grn][item][set] = data[item][set][0][1][1]
                                group[grn][item].append(set)
                else:                # Molecules...
                    ref[grn][title] = {}
                    group[grn][title] = []
                    for set in data[title].keys():
                        if data[title][set][0][0].get():
                            ref[grn][title][set] = data[title][set][0][1][1]
                            try:             group[grn][title].append(set)
                            except KeyError: group[grn][title] = [set]
            for key in group[grn].keys():
                if group[grn][key][0] == 'None':
                    del group[grn][key]
                    del ref[grn][key]
                elif group[grn][key][0] == 'All':
                    group[grn][key] = ['*']
                    ref[grn][key] = {'*':ref[grn][key]['All']}
            if len(group[grn]) == 0:
                group[grn] = None
                ref[grn] = None
                
        return group,ref

    
    def check4all(self,var_list):
        
        for k in var_list.keys():
            try:
                hit = var_list[k].index('All')
                del var_list[k][hit]
            except ValueError: pass
        return var_list


##################################
#                                #
#   Action from MAIN WINDOW      #
#                                #
##################################

    def AtomSelection(self):
        
        frameAtomSel = Toplevel()
        frameAtomSel.title('Atom selection')
        frameAtomSel.resizable(width=NO,height=NO)
        frameAtomSel.initial_focus = frameAtomSel
        f0 = Frame(frameAtomSel,bd=2,relief='groove')
        f0.pack(side=TOP,padx=3,pady=3,expand=YES,fill=BOTH)
        Label(f0, text='Read from PDB file', anchor=W).pack(side=TOP,anchor=W)
        pdbfile = FilenameEntry(f0, browse_pattern = '*.pdb')
        pdbfile.pack(side=TOP, anchor=W)

        item_list = self.types.keys()
        item_list.sort()
        for k in item_list:
            mol = self.types[k][0]
            if mol.__class__.__name__ == 'Protein':
                n   = add.reduce(map(len,mol))
                t1  = k
                t2  = str(n)+' residues'
                f00 = Frame(frameAtomSel,bd=2,relief='groove')
                f00.pack(side=TOP,padx=3,pady=3,fill=BOTH)
                strings = ['All','None','Methyl','BackBone',\
                           'SideChain','Hydrogen','C_alpha','Carbon','Oxygen',
                           'Nitrogen']
                for atom in mol.atomList():
                    if atom.symbol=='S':
                        strings.append('Sulfur')
                        break
                info = {'title': t1, 'natom': len(mol.atomList()),
                        'nres': n, 'deuter': 0}
                Button(f00,text=t1+' : '+t2,command=(lambda self=self,
                       info=info,strings=strings: self.atomsWindow(info,
                       strings,6))).pack(side=TOP,fill=BOTH,expand=YES)
            else:
                f00 = Frame(frameAtomSel,bd=2,relief='groove')
                f00.pack(side=TOP,padx=3,pady=3,fill=BOTH)
                t1  = k
                t2  = str(len(self.types[k]))+' molecules'
                strings = ['All','None']
                met,carb,hydr,oxy,nitr,sulf,phos = getAtoms(mol)
                dcut = 2
                if hydr == 1:
                    strings.append('Hydrogen')
                    dcut = 3
                if met  == 1: strings.append('Methyl')
                if carb == 1: strings.append('Carbon')
                if oxy  == 1: strings.append('Oxygen')
                if nitr == 1: strings.append('Nitrogen')
                if sulf == 1: strings.append('Sulfur')
                if phos == 1: strings.append('Phosphorus')
                if hasattr(mol, 'groups') and len(mol.groups) > 0:
                # deuter is off,special handling needed
                    for name in mol.groups: strings.append(name.name)
                info = {'title': t1, 'natom': len(mol.atomList())*\
                        len(self.types[k]),'nmol': len(self.types[k]),
                        'deuter': 0}
                if len(strings) < 4 and strings.count('Hydrogen') == 0:
                    del strings[2:]
                    dcut = 2
                Button(f00,text=t1+' : '+t2,command=(lambda self=self,
                       dcut=dcut,info=info,
                       strings=strings: self.atomsWindow(info,strings,
                       dcut))).pack(side=TOP,fill=BOTH,expand=YES)

        f2 = Frame(frameAtomSel,bd=2,relief='groove')
        f2.pack(side=TOP,fill=X,padx=3,pady=3)
        Button1=Button(f2,text='OK',
                       command=frameAtomSel.destroy,
                       underline=0)
        Button1.pack(padx=1,pady=1,side=LEFT)

        frameAtomSel.grab_set()
        frameAtomSel.initial_focus.focus_set()
        frameAtomSel.wait_window(frameAtomSel)

        pdbfile = pdbfile.get()
        if pdbfile != '':
            self.readPDBAtomSelection(pdbfile)


    def atomsWindow(self,info,strings,up_limit):

        atomsTL = Toplevel(self)
        atomsTL.title(info['title'])
        atomsTL.resizable(width=NO,height=NO)
        atomsTL.initial_focus = atomsTL
        f0 = Frame(atomsTL,bd=2,relief='flat')
        f0.pack(side=TOP,padx=3,pady=3)
        Label(f0,text=info['title']).grid(column=0,row=0,columnspan=2)
        try:
            Label(f0,text=str(info['nres'])).grid(column=0,row=1)
            Label(f0,text='Residues').grid(column=1,row=1)
        except:
            Label(f0,text=str(info['nmol'])).grid(column=0,row=1)
            Label(f0,text='Molecules').grid(column=1,row=1)
        Label(f0,text=str(info['natom'])).grid(column=0,row=2)
        Label(f0,text='Atoms').grid(column=1,row=2)

        f1 = Frame(atomsTL,bd=2,relief='groove')
        f1.pack(side=TOP,padx=3,pady=3)
        Label(f1,text='Deuter').grid(column=1,row=0)
        Label(f1,text='Selection').grid(column=2,row=0)

        try: abc = self.atomButtonCollection[info['title']]
        except: abc = {'All':[[IntVar(),IntVar()]]}
        gj = Checkbutton(f1,variable=abc['All'][0][0],
                         command=(lambda self=self,where=abc:\
                                  self.checkAtomButton({'All':[2,None]},where,
                                                   which=0,how=0)))
        try: abc['All'][1] = gj
        except IndexError: abc['All'].append(gj)
        gj = Checkbutton(f1,variable=abc['All'][0][1],
                    command=(lambda self=self,where=abc:\
                             self.checkAtomButton({'All':[None,2]},where,
                                              which=1,how=0)))
        try: abc['All'][2] = gj
        except IndexError: abc['All'].append(gj)
        Label(f1,text='All').grid(column=2,row=1)
        
        try: gj = abc['None'][0][0].get()
        except: abc['None'] = [[IntVar(),IntVar()]]
        gj = Checkbutton(f1,variable=abc['None'][0][0],
                         command=(lambda self=self,where=abc:\
                                  self.checkAtomButton({'None':[2,None]},where,
                                                   which=0,how=0)))
        try: abc['None'][1] = gj
        except IndexError: abc['None'].append(gj)
        gj = Checkbutton(f1,variable=abc['None'][0][1],
                    command=(lambda self=self,where=abc:\
                             self.checkAtomButton({'None':[None,2]},where,
                                              which=1,how=0)))
        try: abc['None'][2] = gj
        except IndexError: abc['None'].append(gj)
        Label(f1,text='None').grid(column=2,row=2)
        
        for i in range(2,len(strings)):
            try: gj = abc[strings[i]][0][0].get()
            except: abc[strings[i]] = [[IntVar(),IntVar()]]
            what = {'All':[0,None],'None':[0,None],strings[i]:[2,None]}
            if strings[i] == 'C_alpha': what['BackBone'] = [0,None]
            gj = Checkbutton(f1,variable=abc[strings[i]][0][0],
                             command=(lambda self=self,where=abc,
                             what=what: self.checkAtomButton(what,
                             where,which=0)))
            try: abc[strings[i]][1] = gj
            except IndexError: abc[strings[i]].append(gj)
            if i < up_limit:
                gj = Checkbutton(f1,variable=abc[strings[i]][0][1],
                                 command=(lambda self=self,where=abc,
                                          current=strings[i]:\
                                          self.checkAtomButton({'All':[None,0],
                                          'None':[None,0],current:[None,2]},
                                          where,which=1)))
                try: abc[strings[i]][2] = gj
                except IndexError: abc[strings[i]].append(gj)
            Label(f1,text=strings[i]).grid(column=2,row=i+1)
        for i in range(len(strings)):
            for ia in range(1,len(abc[strings[i]])):
                abc[strings[i]][ia].grid(column=ia-1,row=i+1)
            if i == 0:
                try: gj = self.atomButtonCollection[info['title']]
                except: abc['All'][1].invoke()
            elif i == 1:
                try: gj = self.atomButtonCollection[info['title']]
                except: abc['None'][2].invoke()
        self.atomButtonCollection[info['title']] = abc

        f2 = Frame(atomsTL,bd=2,relief='groove')
        f2.pack(side=TOP,fill=X,padx=3,pady=3)
        Button1=Button(f2,text='Close',
                       command=atomsTL.destroy,
                       underline=0)
        Button1.pack(padx=1,pady=1,side=RIGHT)

        atomsTL.grab_set()
        atomsTL.initial_focus.focus_set()
        atomsTL.wait_window(atomsTL)


    def checkAtomButton(self,what,where,which=None,how=None):
        """ An auxilliary function is a part of Atom Selection system.
        It provides a support for setting up dependencies between buttons
        in a multi-selection widget (e.g. turn off all buttons after pressing
        ALL or NONE, or turn off BackBone after pressing C_ALPHAS).
        what is a dictionary consisting of names of buttons (as keys) and
        a two-element list having either None, 0 (off), 1 (on) or 2 (ignore);
        where is a dictionary with buttons widgets and variables which defines
        their current state (on/off);
        which is a number defining which column of buttons is to be modified;
        how defines a way in which button state is to be modified"""
        
        res = {}
        for ia in what.keys():
            res[ia] = what[ia]
        for i in where.keys():
            if res.has_key(i): pass
            else:
                res[i] = [None,None]
                if how is None: pass
                else:
                    if which is None: res[i] = [how,how]
                    else: res[i][which] = how
        for i in res.keys():
            if res[i][0] is not None:
                if res[i][0] == 2: pass
                else: where[i][0][0].set(res[i][0])
            if res[i][1] is not None:
                if res[i][1] == 2: pass
                else: where[i][0][1].set(res[i][1])
        #
        # default settings
        #
        for i in where.keys():
            gj = where[i][0][0].get()
            if gj: break
        if gj == 0: where['All'][0][0].set(1)
        for i in where.keys():
            gj = where[i][0][1].get()
            if gj: break
        if gj == 0: where['None'][0][1].set(1)


    def readPDBAtomSelection(self, filename):
        """ Read a PDB file for an atom colection (atoms labelled with
        an asterix (*). Assignment of scattering weights for united atoms
        is also possible (e.g. via pdb_atom.properties). """

        self.status.set("Processing PDB file ...")
        pdb = PDBConfiguration(filename)
        selection = []
        total = 0
        for object in pdb.objects:
            if dir(object).count('atom_list') == 1:  # no groups
                for atom in object.atom_list:
                    total = total+1
                    if atom.properties['element'] == '*':
                        selection.append(atom)
            elif dir(object).count('residues') == 1:  # e.g. protein
                for residue in object.residues:
                    for atom in residue.atom_list:
                        total = total+1
                        if atom.properties['element'] == '*':
                            selection.append(atom)
        if len(self.traj.universe.atomList()) != total:
            Dialog.Dialog(self, title='Error',
                          text='File '+filename+' is not a valid PDB file',
                          bitmap='error', default=0,
                          strings = ('Cancel',))
        else:
            Dialog.Dialog(self, title='Information',
                          text=str(len(selection))+' atoms have been'+\
                          ' selected', bitmap='error', default=0,
                          strings = ('Cancel',))
            self.atomsFromPDB = (pdb,selection,filename)
        self.status.clear()
                                

    def GroupSelection(self,var):

        self.groupNumber.set(var)
        frameGroupSel = Toplevel()
        frameGroupSel.title('Groups...')
        frameGroupSel.resizable(width=NO,height=NO)
        frameGroupSel.initial_focus = frameGroupSel
        Label(frameGroupSel,text='Selection for group %s'%var).\
                                            pack(side=TOP,pady=5)

        item_list = self.types.keys()
        item_list.sort()
        for k in item_list:
            mol = self.types[k][0]
            if mol.__class__.__name__ == 'Protein':
                n   = add.reduce(map(len,mol))
                t1  = k
                t2  = str(n)+' residues'
                patterns = getResidues(mol)
                f00 = Frame(frameGroupSel,bd=2,relief='groove')
                f00.pack(side=TOP, padx=3, pady=3,fill=BOTH)
                mb_prot = Menubutton(f00,text=t1+' : '+t2,relief='groove',bd=3)
                menu_prot = Menu(mb_prot,tearoff=0)
                gj = self.groupNumber.get()
                try: self.menuProtein[gj][k].get()
                except:
                    self.menuProtein[gj][k] = StringVar()
                    self.menuProtein[gj][k].set('All')
                menu_prot.add_radiobutton(label='Protein',value='All',
                          variable=self.menuProtein[gj][k],command=\
                         (lambda info={'name':'All','master':k},
                          self=self,pattern={'All':[mol]},
                          root=frameGroupSel: \
                          self.groupsWindow(info,root,pattern=pattern)))
                menu_prot.add_radiobutton(label='SideChain',
                                          value='SideChain',
                          variable=self.menuProtein[gj][k],command=\
                         (lambda info={'name':'SideChain','master':k},
                          self=self,strings=patterns.keys(),
                          root=frameGroupSel,pattern=patterns: \
                          self.groupsWindow(info,root,strings,pattern)))

                try: pat_bb = self.backbone[k]
                except:
                    self.status.set("Processing protein backbone atoms...")
                    pat_bb = getProteinBackbone(mol)
                    self.status.clear()
                    self.backbone[k] = pat_bb
                menu_prot.add_radiobutton(label='BackBone',value='BackBone',
                          variable=self.menuProtein[gj][k],command=\
                         (lambda info={'name':'BackBone','master':k},
                          self=self,pattern={'All':pat_bb},
                          root=frameGroupSel: \
                          self.groupsWindow(info,root,pattern=pattern)))
                try: pat_ch3 = self.methyl[k]
                except:
                    self.status.set("Looking for methyl groups...")
                    pat_ch3 = getMethyls(mol)
                    self.status.clear()
                    self.methyl[k] = pat_ch3
                if len(pat_ch3) > 0:
                    menu_prot.add_radiobutton(label='Methyl',value='Methyl',
                          variable=self.menuProtein[gj][k],command=\
                         (lambda info={'name':'Methyl','master':k}, self=self,
                          pattern={'All':pat_ch3}, root=frameGroupSel: \
                          self.groupsWindow(info,root,pattern=pattern)))
                mb_prot.config(menu=menu_prot)
                mb_prot.pack(side=TOP,expand=YES,fill=BOTH)
            else:
                f00  = Frame(frameGroupSel,bd=2,relief='groove')
                f00.pack(side=TOP, padx=3, pady=3,fill=BOTH)
                t1   = k
                t2   = str(len(self.types[k]))+' molecules'
                # check for presence of Methyl groups...
                Button(f00,text=t1+' : '+t2,command=(lambda self=self,
                          pattern={'All':self.types[k].objects},
                          root=frameGroupSel,info={'name':'','master':k}:\
                          self.groupsWindow(info,root,pattern=pattern))).\
                          pack(side=TOP,expand=YES,fill=BOTH)

        f2 = Frame(frameGroupSel,bd=2,relief='groove')
        f2.pack(side=TOP,fill=X,padx=3,pady=3)
        Button1=Button(f2,text='Close',
                       command=frameGroupSel.destroy,
                       underline=0)
        Button1.pack(padx=1,pady=1,side=RIGHT)

        frameGroupSel.grab_set()
        frameGroupSel.initial_focus.focus_set()
        frameGroupSel.wait_window(frameGroupSel)

                
    def groupsWindow(self,info,root,strings=[],pattern={},menu=None):
        """ info is a dictionary consisting name (e.g. Protein.184,...)
        and other informations (residue number, atom number),
        strings is a list containing names (e.g. lysine,...; [] in case
        of molecules), pattern is a dictionary with keys matching strings,
        which contains MMTK collection od atoms that are going to be match
        """
        groupsTL = Toplevel(root)
        groupsTL.title('Selection')
        groupsTL.resizable(width=NO,height=NO)
        groupsTL.initial_focus = groupsTL
        f0 = Frame(groupsTL,bd=2,relief='flat')
        f0.pack(side=TOP,padx=3,pady=3)
        Label(f0,text=info['name']+' in '+info['master']).pack(side=TOP)
        f1 = Frame(f0,bd=2,relief='groove')
        f1.pack(side=TOP,padx=3,pady=3)
        scroll = Scrollbar(f1)
        c1 = Canvas(f1,bd=2,relief='flat',width=1,height=1,
                    yscrollcommand=scroll.set)
        scroll.config(command=c1.yview)
        scroll.pack(side=RIGHT,fill=Y)
        c1.pack(side=LEFT,expand=YES,fill=BOTH,padx=3,pady=3)

        fc = Frame(c1)
        Label(fc,text='Name').grid(column=1,row=0)
        Label(fc,text='Reference').grid(column=2,row=0)
        Label(fc,text='Info').grid(column=3,row=0)

        # gbc: [[on/off,PDB-filename,info],button-selection,b-reference,b-info]
        title = info['master']+' '+info['name']
        title = strip(title)
        try: gbc = self.groupButtonCollection[self.groupNumber.get()][title]
        except: gbc = {'All':[[IntVar(),['None',None],None],None,None,None]}
        gbc['All'][1] = Checkbutton(fc,variable=gbc['All'][0][0],text='All',
                         command=(lambda self=self,where=gbc:\
                         self.checkGroupButton({'All':2},where,how=0)))
        gbc['All'][1].grid(column=0,row=1,columnspan=2,sticky=W)
        if pattern.has_key('All'):
            gbc['All'][2] = Button(fc,text=gbc['All'][0][1][0],
                command=(lambda self=self,match=pattern['All'],master=title,
                item='All',root=groupsTL: self.readGroupReference(match,
                master,item,root)))
            gbc['All'][2].grid(column=2,row=1)
            gbc['All'][3] = Button(fc,text='Info',command=(lambda self=self,
                master=title,root=groupsTL: \
                self.groupsInfo(master,'All',root)))
            gbc['All'][3].grid(column=3,row=1)
        
        try: gj = gbc['None'][0][0].get()
        except: gbc['None'] = [[IntVar(),[None,None],None],None,None,None]
        gbc['None'][1] = Checkbutton(fc,variable=gbc['None'][0][0],text='None',
                         command=(lambda self=self,where=gbc:\
                         self.checkGroupButton({'None':2},where,how=0)))
        gbc['None'][1].grid(column=0,row=2,columnspan=2,sticky=W)

        if len(strings) > 0: 
            for i in range(len(strings)):
                try: gj = gbc[strings[i]][0][0].get()
                except: gbc[strings[i]] = [[IntVar(),['None',None],None],
                                           None,None,None]
                what = {'All':0,'None':0,strings[i]:2}
                gbc[strings[i]][1] = Checkbutton(fc,
                   variable=gbc[strings[i]][0][0],command=(lambda self=self,
                   where=gbc,what=what: self.checkGroupButton(what,where)),
                   text=strings[i])
                gbc[strings[i]][1].grid(column=0,row=i+3,columnspan=2,sticky=W)
                gbc[strings[i]][2] = Button(fc,text=gbc[strings[i]][0][1][0],
                   command=(lambda self=self,match=pattern[strings[i]],
                   master=title, item=strings[i],root=groupsTL: \
                   self.readGroupReference(match,master,item,root)))
                gbc[strings[i]][2].grid(column=2,row=i+3)
                gbc[strings[i]][3] = Button(fc,text='Info',
                   command=(lambda self=self,master=title,
                   item=strings[i],root=groupsTL: \
                   self.groupsInfo(master,item,root)))
                gbc[strings[i]][3].grid(column=3,row=i+3)
                if gbc[strings[i]][0][0].get() == 0:
                    gbc[strings[i]][2].config(state=DISABLED)
                    gbc[strings[i]][3].config(state=DISABLED)

        try: gj = self.groupButtonCollection[self.groupNumber.get()]\
                  [title]['All'][0][0].get()
        except: gbc['All'][1].invoke()
        if gbc['All'][0][0].get() == 0:
            try:
                gbc['All'][2].config(state=DISABLED)
                gbc['All'][3].config(state=DISABLED)
            except: pass
        self.groupButtonCollection[self.groupNumber.get()][title] = gbc
        fc.update_idletasks()
        fc_h = fc.winfo_reqheight()
        fc_w = fc.winfo_reqwidth()
        c1.create_window(0,0,window=fc,anchor=NW)
        c1.config(height=min(200,fc_h),width=fc_w,scrollregion=(0,0,fc_w,fc_h))
        if fc_h < 200: scroll.pack_forget()

        f2 = Frame(groupsTL,bd=2,relief='groove')
        f2.pack(side=TOP,fill=X,padx=3,pady=3)
        Button1=Button(f2,text='Close',
                       command=groupsTL.destroy,
                       underline=0)
        Button1.pack(padx=1,pady=1,side=RIGHT)

        groupsTL.grab_set()
        groupsTL.initial_focus.focus_set()
        groupsTL.wait_window(groupsTL)
        try: root.grab_set()
        except: pass


    def groupsInfo(self,master,item,root):
        """ show info about the number of residues of given type
        and which atoms were matched """

        data = self.groupButtonCollection[self.groupNumber.get()]\
                    [master][item][0][2]
        if data:
            filename = self.groupButtonCollection[self.groupNumber.get()]\
                       [master][item][0][1][1]
            ifil = rfind(filename,'/')
            if ifil > 0: filename = filename[ifil+1:]
            gi = Toplevel(root)
            gi.resizable(width=NO,height=NO)
            gi.title('Selection Info')
            f0 = Frame(gi,bd=2,relief='groove')
            f0.pack(side=TOP,padx=3,pady=3)
            natom = 0
            for i in data['PDB']: natom = natom + len(i.atomList())
            Label(f0,text='PDB reference is: %s'%filename)\
                  .pack(side=TOP,pady=5)
            Label(f0,text='Number of objects: %d'%len(data['MMTK']),
                  fg='#FF0000').pack(side=TOP,pady=5)
            Label(f0,text='Number of atoms\nmatching PDB: %d' % natom,
                  fg='#FF0000').pack(side=TOP,pady=5)
            try:
                gi.grab_set()
                gi.wait_window(gi)
                root.grab_set()
            except: pass


    def checkGroupButton(self,what,where,how=None):

        res = {}
        for ia in what.keys(): res[ia] = what[ia]
        for i in where.keys():
            if res.has_key(i): pass
            else: res[i] = how
        for i in res.keys():
            if res[i] is not None:
                if res[i] == 2: pass
                else: where[i][0][0].set(res[i])
        for i in where.keys():
            gj = where[i][0][0].get()
            if gj: break
        if gj == 0:
            where['All'][0][0].set(1)
            try:
                where['All'][2].config(state=NORMAL)
                where['All'][3].config(state=NORMAL)
            except: pass
        else:
            for i in where.keys():
                if i != 'All' and i != 'None':
                    if where[i][0][0].get() == 1:
                        where[i][2].config(state=NORMAL)
                        where[i][3].config(state=NORMAL)
                    else:
                        where[i][2].config(state=DISABLED)
                        where[i][3].config(state=DISABLED)
        if where['All'][0][0].get() or where['None'][0][0].get():
            for i in where.keys():
                if i != 'All' and i != 'None':
                    where[i][2].config(state=DISABLED)
                    where[i][3].config(state=DISABLED)
        if where['All'][0][0].get() == 0:
            try:
                where['All'][2].config(state=DISABLED)
                where['All'][3].config(state=DISABLED)
            except: pass
        else:
            try:
                where['All'][2].config(state=NORMAL)
                where['All'][3].config(state=NORMAL)
            except: pass


    def readGroupReference(self,pattern,master,item,root):

        fd       = FileDialog.LoadFileDialog(root)
        filename = fd.go(key='LoadPDB', pattern='*.pdb')
        self.status.set("Processing PDB file ...")
        if filename:
            gj = parsePDBReference(filename,pattern,verbose=self.verbose.get())
            if gj is None:
                self.groupButtonCollection[self.groupNumber.get()]\
                                          [master][item][0][1][1] = None
                self.groupButtonCollection[self.groupNumber.get()]\
                                   [master][item][2].config(text='None')
                self.groupButtonCollection[self.groupNumber.get()]\
                                   [master][item][0][1][0] = 'None'
            else:
                info = gj
                self.groupButtonCollection[self.groupNumber.get()][master]\
                                      [item][0][1][1] = filename
                self.groupButtonCollection[self.groupNumber.get()][master]\
                                      [item][2].config(text='Set')
                self.groupButtonCollection[self.groupNumber.get()][master]\
                                      [item][0][1][0] = 'Set'
                self.groupButtonCollection[self.groupNumber.get()][master]\
                                      [item][0][2] = info
        self.status.clear()
        root.grab_set()
#######################################################
         
    def ViewData(self, category):
        self.plotlist = []
        self.selection = None
        variable_names = self.categories[category][:]
        if category == 'configuration':
            Dialog.Dialog(self, title='Undefined operation',
                      text='This operation is not permited.',
                      bitmap='warning', default=0,
                      strings = ('Cancel',))
        elif category == 'energy':
            EnergyViewer(self, string.capitalize(category),
                         self.time, self.traj, variable_names)
        else:
            PlotViewer(self, string.capitalize(category),
                       self.time, self.traj, variable_names)

    def _selectRange(self, range):
        if range is None:
            first = 0
            last = len(self.traj)
            self.selection = None
        else:
            first = sum(less(self.time, range[0]))
            last = sum(less(self.time, range[1]))
            last = min(last, len(self.traj)-1)
            self.selection = (first, last)
        self.e41.set(first)
        self.e42.set(last)
        self.getTime()
        self._updateTimeInfo()
        for plot_canvas in self.plotlist:
            if range is None:
                plot_canvas.select(None)
            else:
                plot_canvas.select((self.time[first], self.time[last]))

    def _registerPlot(self, plot_canvas):
        self.plotlist.append(plot_canvas)

    def _unregisterPlot(self, plot_canvas):
        try:
            self.plotlist.remove(plot_canvas)
        except ValueError:
            pass

    def info(self):
        info = Toplevel()
        info.title('Trajectory Information')
        info.resizable(width=NO,height=NO)
        info.initial_focus = info
        f0 = Frame(info,bd=2,relief='groove')
        f0.pack(side=TOP,padx=3,pady=3,fill=BOTH)
        try:
            info_string = self.trajInfo
        except AttributeError:
            info_string = 'No trajectory loaded'
        Label(f0,text=info_string).pack(side=TOP,fill=BOTH)
        f1 = Frame(info,bd=2,relief='groove')
        f1.pack(side=TOP,fill=X,padx=3,pady=3)
        Button1=Button(f1,text='Close',
                       command=info.destroy,
                       underline=0)
        Button1.pack(padx=1,pady=1,side=RIGHT)
        info.grab_set()
        info.initial_focus.focus_set()
        info.wait_window(info)

###############################################################################

    def gj(self):
        Dialog.Dialog(self, title='Error',
                      text = 'Problems with your settings occured. '+\
                      'Try solve them and run calculations one more time.', 
                      bitmap='error', default=0, strings = ('OK',))

class fancyDialog(Toplevel):

    def __init__(self,parent,title,inList,switch):

        Toplevel.__init__(self,parent)
        self.transient(parent)
        self.resizable(width=NO,height=NO)
        self.title(title)
        self.cmd = None 
        self.parent = parent
        frame = Frame(self,bd=2,relief='flat')
        self.initial_focus = self
        frame.pack(side=TOP)
        # moze tylko Label...
        textA = Text(self,relief='flat',padx=10,pady=5,setgrid=1,width=40,
                     wrap='word',exportselection=0,font=parent.myFont,height=5)
        textA.tag_config("b",foreground='black')
        textA.tag_config("r",foreground='red')
        textA.insert(END,"Your settings will stored in an input file"+\
                         " whose contents are shown below. You can ",
                     ("b",))
        textA.insert(END,"Save",("r",))
        textA.insert(END," them and run the calculations later or ",("b",))
        textA.insert(END,"Run",("r",))
        textA.insert(END," the calculations immediately.",("b",))
        textA.configure(state=DISABLED)
        textA.pack(side=TOP)

        finp = Frame(self,bd=2,relief='flat')
        finp.pack(side=TOP)
        textB = Text(finp,relief='ridge',padx=10,pady=5,setgrid=1,width=60,
                     height=10,wrap='none')
        mystr = ""
        hbarFlag = 0
        for i in inList:
            for ia in range(len(i)):
                mystr = mystr+" "+str(i[ia])
            mystr = mystr+"\n"
            if len(mystr) > 60: hbarFlag = 1
        textB.insert(END,mystr)
        textB.configure(state=DISABLED)
        textB.pack(side=TOP)
        if len(inList) > 10:
            vbar = Scrollbar(finp,name='vbar')
            vbar.pack(side=RIGHT,fill=Y)
            textB.configure(yscrollcommand=vbar.set)
            vbar.configure(command=textB.yview)
        if hbarFlag:
            hbar = Scrollbar(finp,name='hbar',orient='horizontal')
            hbar.pack(side=BOTTOM,fill=X)
            textB.configure(xscrollcommand=hbar.set)
            hbar.configure(command=textB.xview)
        Button(self,text="Save",command=(lambda self=self,\
               input=inList: self.saveInputFile(input))).pack(side=LEFT)
        Button(self,text='Run',command=(lambda self=self,\
               input=inList,switch=switch:\
               self.runCalc(input,switch))).pack(side=LEFT,expand=YES)
        Button(self,text='Cancel',command=self.cancel)\
               .pack(side=RIGHT)
        if mac_conventions:
            self.bind('<Escape>', lambda event: self.cancel())
        else:
            self.bind('<Cancel>', lambda event: self.cancel())
        self.grab_set()
        if not self.initial_focus: self.initial_focus = self
        self.protocol("WM_DELETE_WINDOW",self.cancel)
        self.initial_focus.focus_set()
        self.wait_window(self)


    def cancel(self,event=None):

        self.parent.focus_set()
        self.destroy()


    def saveInputFile(self,data):

        self.withdraw()
        self.update_idletasks()
        data.insert(0,["from MMTK import *"])
        saveText(master=self,data=data)
        self.cancel()


    def runCalc(self,input,switch):

        self.withdraw()
        self.update_idletasks()
        filename = saveText(filename=mktemp(),data=input)
        text = "The following command is about to be run in the background:"+\
               "\n\n"
        cmd  = self.parent.command+switch+filename+' 1> '+\
               self.parent.LogFilename+' 2>&1 &'
        go = askokcancel("Running...",text+cmd)
        if not go:
            cmd = None
        self.cmd = cmd
        self.cancel()


#########################
#                       #
#    Main Program       #
#                       #
#########################

root=Tk()

try:
    root.tk.call('console','hide')
except TclError:
    pass

if sys.platform == 'darwin':
    mac_conventions = 'X11' not in root.winfo_server()
    try:
        root.tk.call('package', 'require', 'tile')
        root.tk.call('namespace', 'import', '-force', 'ttk::*')
        root.tk.call('tile::setTheme', 'aqua')
    except TclError:
        pass
else:
    mac_conventions = 0

root.title('nMOLDYN')
root.resizable(width=NO,height=NO)
xMOLDYN = xMOLDYN(root)
if len(sys.argv) == 2:
    xMOLDYN.loadTrajectory(sys.argv[1])
root.mainloop()
