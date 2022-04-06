import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setuptools.setup(
    name='mlmodelwatermarking',
    version='0.0.1',
    author='SAP SE',
    maintainer='Sofiane Lounici',
    maintainer_email='sofiane.lounici@sap.com',
    description='Ml Model Watermarking',
    install_requires=requirements(),
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/SAP/ml-model-watermarking',
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Developers',
        'Development Status :: 3 - Alpha
    ],
    python_requires='>3.5, <3.10',
    keywords='watermarking SAP',
)
