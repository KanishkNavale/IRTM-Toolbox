import setuptools

# Readme.md Handler
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='irtm',
                 version='0.0.1',
                 description='A toolbox for Information Retreival & Text Mining.',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='https://github.com/KanishkNavale/IRTM-Toolbox.git',
                 author='Kanishk Navale',
                 author_email='navalekanishk@gmail.com',
                 license='MIT',
                 package_dir={"": "src"},
                 packages=setuptools.find_packages(where="src"),
                 install_requires=[
                                    'nltk==3.5'
                                  ],
                 include_package_data=True,
                 zip_safe=False,
                 )
