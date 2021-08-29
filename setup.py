import setuptools


# Readme.md Handler
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
                 # Package Name
                 name='irtm',

                 # Version ID
                 version='0.0.1',

                 # Package Descriptions
                 description='A toolbox for Information Retreival & Text Mining.',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='https://github.com/KanishkNavale/IRTM-Toolbox.git',

                 # Developer Details
                 author='Kanishk Navale',
                 author_email='navalekanishk@gmail.com',

                 # License
                 license='MIT',

                 # Package Build Source Pointer
                 package_dir={"": "src"},
                 packages=setuptools.find_packages(where="src"),

                 # Setup Prequisites
                 install_requires=[
                                    'nltk==3.5'
                                  ],

                 # WrapUP
                 include_package_data=True,
                 zip_safe=False,
                 )
