from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("VERSION", "r", encoding="utf-8") as f:
    version = f.read()

setup(
    name='wlcoref',
    version=version,
    description='Packaged code from the Word-Level Coreference Resolution paper by V. Dobrovolskii, 2021',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ispras-texterra/wl-coref',
    author='ISPRAS MODIS NLP team',
    author_email='modis@ispras.ru',
    maintainer='Sasha Pivovarov',
    maintainer_email='pivovarov.av@ispras.ru',
    packages=find_packages(include=['coref']),
    install_requires=['jsonlines', 'toml', 'transformers==3.2.0', 'torch==1.4.0', 'torchvision==0.5.0'],
    data_files=[('', ['VERSION'])],
    python_requires='>=3.6',
    license='Not stated',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ]
)
