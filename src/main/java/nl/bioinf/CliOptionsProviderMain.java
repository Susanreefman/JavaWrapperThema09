package nl.bioinf;

import java.util.Arrays;

/**
 * CliOptionsProviderMain - main to execute program
 */
public class CliOptionsProviderMain {
    /**
     * main - main method to start parsing input and algorithm
     * @param args
     */
    public static void main(String[] args) {
        System.out.println("Arrays.toString(args) = "
                + Arrays.toString(args));
        OptionsProvider optionsProvider = new CliOptionsProvider(args);
        InputAlgorithm inputAlgorithm = new InputAlgorithm();
        inputAlgorithm.start(optionsProvider);
    }
}

