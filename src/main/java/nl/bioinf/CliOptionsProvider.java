package nl.bioinf;

import org.apache.commons.cli.*;

/**
 * CliOptionsProvider - parse options input arguments
 */
public class CliOptionsProvider implements OptionsProvider{
    private Options options;
    private CommandLine cmd;
    private String inputFile;
    private String outputFile;

    /**
     * CliOptionsProvider - starts parser
     * @param args
     */
    public CliOptionsProvider(String[] args) {
        init();
        parseArgs(args);
    }

    /**
     * parseArgs - argument parser if verified, if input is option 'h' print help
     * @param args
     */
    private void parseArgs(String[] args) {
        CommandLineParser parser = new DefaultParser();
        try {
            this.cmd = parser.parse(options, args);
            if (cmd.hasOption('h')) {
                printHelp();
            }
            verify();
        } catch (ParseException e) {
            System.out.println("Something went wrong while parsing, cause: "
                    + e.getMessage());
            printHelp();
        }
    }

    /**
     * printHelp - help formatter, printing help options
     */
    private void printHelp() {
        HelpFormatter formatter = new HelpFormatter();
        formatter.printHelp( "java -jar MyApp.jar [options]", options );
    }

    /**
     * init - input argument options
     */
    private void init(){
        this.options = new Options();
        options.addOption(new Option("i",
                "input",
                true,
                "The input file - expects numeric columns, no missing values"));
        options.addOption(new Option("o",
                "output",
                false,
                "The input file - with unclassified instances"));
        options.addOption(new Option("h",
                "help",
                false,
                "Prints the help"));
    }

    /**
     * verify - verifying input arguments
     * @throws ParseException - if no file provided
     */
    private void verify() throws ParseException {
        // verifying input file
        if (!cmd.hasOption('i')) {
            throw new ParseException("no file provided");
        } else {
            this.inputFile = cmd.getOptionValue('i');
        }
        // verifying output file
        if (!cmd.hasOption('o')) {
            throw new ParseException("no file provided");
        } else {
            this.outputFile = cmd.getOptionValue('o');
        }
    }


    @Override
    public String getFileName() { return inputFile; }

}