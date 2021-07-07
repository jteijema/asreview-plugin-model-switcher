from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='asreview-model-switcher-extention',
    version='0.3',
    description='The extention that adds models that switch at a certain set point.',
    url='https://github.com/JTeijema/ASReview-Model_Switcher',
    author='ASReview team, Jelle Teijema',
    author_email='j.j.teijema@gmail.com',
    classifiers=[
        'Development Status :: 3 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='systematic review',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    python_requires='~=3.6',
    install_requires=[
        'sklearn',
        'asreview>=0.13',
        'tensorflow',
        'scipy'
    ],
    entry_points={
        'asreview.models.classifiers': [
            'svm_cnn = asreviewcontrib.models.svm_cnn:SVM_CNN_Model',
            'power_cnn = asreviewcontrib.models.cnn:POWER_CNN',
        ],
        'asreview.models.feature_extraction': [
            # define feature_extraction algorithms
        ],
        'asreview.models.balance': [
            # define balance strategy algorithms
        ],
        'asreview.models.query': [
            # define query strategy algorithms
        ]
    },
    project_urls={
        'Bug Reports': 'https://github.com/JTeijema/ASReview-Model_Switcher/issues',
        'Source': 'https://github.com/JTeijema/ASReview-Model_Switcher',
    },
)
