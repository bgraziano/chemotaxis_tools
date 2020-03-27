from setuptools import setup

files = ['chemotaxis_tools/*']

setup_parameters = dict(
      name='chemotaxis_tools',
      version='0.1',
      description='Functions for analyzing data from cell migration assays',
      url='https://github.com/bgraziano/chemotaxis_tools',
      author='Brian Graziano',
      author_email='brgrazian@gmail.com',
      license='MIT',
      packages=['chemotaxis_tools'],
      install_requires=['numpy>=1.17.2', 'scikit-image>=0.15.0', 'pandas>=1.0.2'],
      include_package_data=True,
      zip_safe=False,
      python_requires='>=3.7',
      )

setup(**setup_parameters)
