from pybuilder.core import init, use_plugin

use_plugin("pypi:pybuilder_research_plugin")
use_plugin("python.core")
use_plugin("python.coverage")
use_plugin("python.distutils")
use_plugin("python.flake8")
use_plugin("python.install_dependencies")
use_plugin("python.integrationtest")
use_plugin("python.pycharm")
use_plugin("python.unittest")

default_task = "publish"
name = "ptcap"


@init
def set_properties(project):
    project.depends_on('pandas')
    project.depends_on('pycocoevalcap',
                       url=('git+ssh://git@github.com/TwentyBN/'
                            'pycocoevalcap.git@sub_packages'))
    project.depends_on('pyfackel')
    project.depends_on('rtorchn', url=('git+ssh://git@github.com/TwentyBN/'
                                       '20bn-rtorchn.git'))
    project.depends_on('tensorboard-pytorch')
    project.depends_on('testfixtures')

    project.set_property('coverage_threshold_warn', 0)
    project.set_property('flake8_verbose_output', "True")
    project.set_property('integrationtest_inherit_environment', True)
