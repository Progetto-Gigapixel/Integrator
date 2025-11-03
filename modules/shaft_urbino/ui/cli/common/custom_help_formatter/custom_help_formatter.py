import argparse

from locales.localization import _


class CustomHelpFormatter(argparse.HelpFormatter):

    def _format_action_invocation(self, action):
        parts = []

        # Handles short and long options separately
        opts = [opt for opt in action.option_strings if len(opt) == 2]  # Short options
        opts_long = [
            opt for opt in action.option_strings if len(opt) > 2
        ]  # Long options

        # Adds information on arguments, defaults, and choices if present
        if action.nargs != 0:
            default = (
                _(" [Default: {default}]").format(default=action.default)
                if action.default is not None
                else ""
            )
            choices = (
                _(" [Choices: {choices}]").format(choices=", ".join(action.choices))
                if action.choices
                else ""
            )
            arg_fmt = action.metavar if action.metavar else action.dest.upper()
            parts.append(
                (_("Parameter: {opts_long} ({opts}){default}")).format(
                    opts_long=", ".join(opts_long),
                    opts=", ".join(opts),
                    default=default,
                )
            )
            parts.append(
                (_("Argument: {arg_fmt}{choices}")).format(
                    arg_fmt=arg_fmt, choices=choices
                )
            )

        return "\n".join(parts)

    def _format_action(self, action):
        # Temporarily stores the original help text and removes it from the action
        original_help = action.help
        action.help = None

        # If the action is the 'help' action, return without formatting
        if action.dest == "help":
            return

        # Calls the base action formatting without the default description
        parts = super(CustomHelpFormatter, self)._format_action(action).split("\n")

        # Restores the original help text to the action
        action.help = original_help

        # Removes extra spaces and organizes the help text
        if parts:
            parts = [part.strip() for part in parts if part.strip()]
            # Adds the custom version of the description
            if original_help:

                description = ("\n\n" + _("Description: {original_help}")).format(
                    original_help=original_help
                )
                # mette in testa description
                parts.insert(0, description)
                # parts.append(description)

        return "\n".join(parts)

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "Usage: "
        # Calls the base class method to maintain consistency with the standard argparse format
        return super(CustomHelpFormatter, self)._format_usage(
            usage, actions, groups, prefix
        )
