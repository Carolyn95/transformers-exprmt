"""  
   Philosophy:
      REFERENCE: https://huggingface.co/docs/datasets/_images/datasets_doc.jpg
      Create a new class inherits `datasets.GeneratorBasedBuilder` class.
      This sub-class needs to override 3 methods:
      {
        `_info`: descriptive information of the dataset, including citation, etc.
                 You could define features used in the dataset here also, 
                 features is an object of `datasets.Features`

        `_split_generators`: returns generator of each split of the dataset

        `_generate_examples`: yield dataset example in each split consuming by DatasetLoader
      }
"""