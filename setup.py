import setuptools


# Readme.md Handler
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
                 # Package Name
                 name='irtm',

                 # Version ID
                 version='0.0.4',

                 # Package Descriptions
                 description='A toolbox for Information Retrieval & Text Mining.',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='https://github.com/KanishkNavale/IRTM-Toolbox.git',

                 # Developer Details
                 author='Kanishk Navale',
                 author_email='navalekanishk@gmail.com',

                 # License
                 license='MIT',

                 # Setup Prequisites
                 install_requires=[
                                    'numpy',
                                    'nltk',
                                    'scikit_learn'
                                  ],

                 # Package Build Source Pointer
                 classifiers=[
                              "Programming Language :: Python :: 3",
                              "License :: OSI Approved :: MIT License",
                              "Operating System :: OS Independent",
                             ],
                 package_dir={"": "src"},
                 packages=setuptools.find_packages(where="src"),
                 python_requires=">=3.6",
                 )
