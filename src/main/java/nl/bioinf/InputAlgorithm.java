package nl.bioinf;


/**
 * InputAlgorithm - Gets the input files and start algorithm
 */
public class InputAlgorithm {
    /**
     * start - Gets input file name and start algorithm
     * @param optionsProvider
     */
     public void start(OptionsProvider optionsProvider) {
         // get input file name
         String inputFile = optionsProvider.getFileName();
         System.out.println("inputFile = " + inputFile);

         // Starts algorithm with input file
         WekaRunner runner = new WekaRunner();
         runner.start(inputFile);
    }
}
