import setuptools

# Readme.md Handler
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='irtm',
                 version='0.0.1',
                 description='This package holds pivotal functions for IRTM.',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='git@github.com:KanishkNavale/IRTM-Toolbox.git',
                 author='Kanishk Navale',
                 author_email='navalekanishk@gmail.com',
                 license='MIT',
                 package_dir={"": "src"},
                 packages=setuptools.find_packages(where="src"),
                 include_package_data=True,
                 zip_safe=False,
                 )
