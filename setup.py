from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='asreview-model-switcher-extention',
    version='0.2',
    description='The extention that adds models that switch at a certain point.',
    url='https://github.com/JTeijema/ASReview-Model_Switcher',
    author='ASReview team, Jelle Teijema',
    author_email='j.j.teijema@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
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
            'SVM_NN = asreviewcontrib.models.SVM_NN:SVM_NN_Model',
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
