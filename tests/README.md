## Testing

This code base contains two types of tests:
- reference tests
- unit tests

The references tests test if the output of the code matches some earlier (reference) output of the code. 
Note that these tests can't assert correctness of the output, they can only assert that the output has been mutated.

The unit tests build small examples to run the various steps on. 
Since the code is not easily testable in it's current form (no composition etc.) so these tests make heavy use of monkey patching so that toy version of e.g. a SkimsSource and SegsSource can be used. 
