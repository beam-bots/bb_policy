# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

if Code.ensure_loaded?(Igniter) do
  defmodule Mix.Tasks.BbPolicy.Install do
    @shortdoc "Installs a BB.Policy command into a project"
    @moduledoc """
    #{@shortdoc}

    Scaffolds a learned-policy command on your robot module: a `command` whose
    handler is `BB.Policy.Command` driving `BB.Policy.ONNX`. The model path,
    observation, and action specs are scaffolded as `TODO`s for you to fill in
    once you know the model and the joint paths in your topology.

    Once filled in, the command is awaitable (`MyRobot.<name>(%{})` +
    `BB.Command.await/2`), governed by the safety system, and usable as a
    `bb_reactor` `command :<name>` step.

    ## Example

    ```bash
    mix igniter.install bb_policy
    mix igniter.install bb_policy --name pick_mug
    ```

    ## Options

    * `--robot` - The robot module (defaults to `{AppPrefix}.Robot`).
    * `--name` - The command name (default `policy`).
    """

    use Igniter.Mix.Task

    alias Igniter.Code.{Common, Function}
    alias Igniter.Project.Formatter

    @impl Igniter.Mix.Task
    def info(_argv, _parent) do
      %Igniter.Mix.Task.Info{
        schema: [robot: :string, name: :string],
        aliases: [r: :robot, n: :name]
      }
    end

    @impl Igniter.Mix.Task
    def igniter(igniter) do
      options = igniter.args.options
      robot_module = BB.Igniter.robot_module(igniter)
      name = options |> Keyword.get(:name, "policy") |> String.to_atom()

      igniter
      |> Formatter.import_dep(:bb_policy)
      |> add_command(robot_module, name, command_body())
      |> Igniter.add_notice(todo_notice(name))
    end

    defp command_body do
      """
      handler {BB.Policy.Command,
        policy: BB.Policy.ONNX,
        policy_opts: [
          model: "priv/models/TODO.onnx",
          observation: [positions: [:TODO]],
          action: [{[:TODO], :position}]
        ],
        rate_hz: 20}
      allowed_states [:idle]
      timeout :timer.seconds(30)
      """
    end

    defp add_command(igniter, robot_module, name, body_code) do
      Spark.Igniter.update_dsl(igniter, robot_module, [{:section, :commands}], nil, fn zipper ->
        if command_exists?(zipper, name) do
          {:ok, zipper}
        else
          code = "command :#{name} do\n#{indent(body_code)}\nend\n"
          {:ok, Common.add_code(zipper, code)}
        end
      end)
    end

    defp command_exists?(zipper, name) do
      case Function.move_to_function_call_in_current_scope(
             zipper,
             :command,
             [2, 3],
             &Function.argument_equals?(&1, 0, name)
           ) do
        {:ok, _} -> true
        _ -> false
      end
    end

    defp indent(text) do
      text
      |> String.split("\n")
      |> Enum.map_join("\n", fn
        "" -> ""
        line -> "  " <> line
      end)
    end

    defp todo_notice(name) do
      """
      bb_policy: a :#{name} command was scaffolded with BB.Policy.ONNX. Replace
      the `TODO` model path, observation joints, and action joints with values
      for your model and topology, and drop the .onnx into priv/models/.
      """
    end
  end
else
  defmodule Mix.Tasks.BbPolicy.Install do
    @shortdoc "Installs a BB.Policy command into a project"
    @moduledoc false
    use Mix.Task

    def run(_argv) do
      Mix.shell().error("""
      The bb_policy.install task requires igniter.

          mix igniter.install bb_policy
      """)

      exit({:shutdown, 1})
    end
  end
end
