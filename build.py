from pybuilder.core import use_plugin, init

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.install_dependencies")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")
use_plugin("python.pycharm")
use_plugin("pypi:pybuilder_research_plugin")

name = "ptcap"
default_task = "publish"


@init
def set_properties(project):
    project.depends_on('pyfackel')
    project.depends_on('pandas')
    project.depends_on('rtorchn', url=('git+ssh://git@github.com/TwentyBN/'
                                       '20bn-rtorchn.git@captioning'))
    project.set_property('flake8_verbose_output', "True")
    project.set_property('coverage_threshold_warn', 0)
