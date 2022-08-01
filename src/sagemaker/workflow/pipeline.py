# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""The Pipeline entity for workflow."""
from __future__ import absolute_import

import json
from collections import defaultdict

from copy import deepcopy
from typing import Any, Dict, List, Sequence, Union, Optional, Set

import attr
import botocore
from botocore.exceptions import ClientError

from sagemaker import s3
from sagemaker._studio import _append_project_tags
from sagemaker.session import Session
from sagemaker.workflow.callback_step import CallbackOutput, CallbackStep
from sagemaker.workflow.lambda_step import LambdaOutput, LambdaStep
from sagemaker.workflow.entities import (
    Entity,
    Expression,
    RequestType,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import (
    Parameter,
    ParameterString,
    ParameterInteger,
    ParameterFloat,
)
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.parallelism_config import ParallelismConfiguration
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import Step
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.utilities import list_to_request, generate_display_edges

STEP_COLORS = {
    "Succeeded": "green",
    "Failed": "red",
    "Executing": "royalblue",
    "Not Executed": "grey",
    "Stopped": "purple",
    "Stopping": "purple",
    "Starting": "royalblue",
}
PARAMETER_TYPE = {"String": ParameterString, "Integer": ParameterInteger, "Float": ParameterFloat}

_NEXT_STEP_NAME = "NextStepName"
_EDGE_LABEL = "EdgeLabel"
_STEP_NAME = "StepName"
_OUT_BOUND_EDGES = "OutBoundEdges"


def load(pipeline_name: str, sagemaker_session: Session = Session()):
    """Loads an existing pipeline based on the pipeline name and returns a Pipeline object

    Note: the steps in the object are left empty

    Args:
        pipeline_name (str): the name for a specific pipeline
        sagemaker_session (sagemaker.session.Session): Session object that manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            pipeline creates one using the default AWS configuration chain.

    Returns:
        Pipeline object
    """
    pipelineEntity = sagemaker_session.sagemaker_client.describe_pipeline(
        PipelineName=pipeline_name
    )
    pipelineDefinition = json.loads(pipelineEntity["PipelineDefinition"])

    parameters = []

    for parameter in pipelineDefinition["Parameters"]:
        parameters.append(
            PARAMETER_TYPE[parameter["Type"]](
                name=parameter["Name"], default_value=parameter.get("DefaultValue", None)
            )
        )

    return ImmutablePipeline(name=pipelineEntity["PipelineName"], parameters=parameters, steps=[])


def list_pipelines(
    sagemaker_session: Session = Session(), max_results: int = 100, next_token: str = None
):
    """Lists all the existing pipelines

    Args:
        sagemaker_session (sagemaker.session.Session): Session object that manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            pipeline creates one using the default AWS configuration chain.
        next_token (str): If the result of the previous call was truncated, a token that can be used to retrieve
            the next set of pipelines

    Returns:
        List of ImmutablePipeline objects
    """
    if next_token is None:
        response = sagemaker_session.sagemaker_client.list_pipelines(MaxResults=max_results)
    else:
        response = sagemaker_session.sagemaker_client.list_pipelines(
            NextToken=next_token, MaxResults=max_results
        )
    pipelineList = []
    pipelineSummaries = response["PipelineSummaries"]
    nextToken = response.get("NextToken", None)

    for pipelineSummary in pipelineSummaries:
        pipelineList.append(load(pipelineSummary["PipelineName"]))

    return _PipelineList(pipelines=pipelineList, next_token=nextToken)


def build_visual_dag(
    pipeline_name: str,
    adjacency_list: List[Dict[str, any]],
    display_edges: Set,
    step_statuses: Dict[str, str],
):
    """Builds a Graphviz object that visualizes a pipeline/execution

    Args:
        pipeline_name (str): pipeline name for the visualized pipeline
        adjacency_list (List[Dict[str, any]]): adjacency list for the visualized pipeline
        display_edges (set): edges to be displayed in the visualized pipeline
        step_statuses (Dict[str, str]): step statuses of the steps in an execution

    Returns:
        A Graphviz object
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            """
            Please install graphviz library to use this method.
            Follow installation instructions outlined on the 'Installation' section of
            https://graphviz.readthedocs.io/en/stable/manual.html.
            NOTE: Verify that Graphviz and Anaconda are installed.
            """
        )

    G = graphviz.Digraph(pipeline_name, strict=True)

    for step in adjacency_list:
        parent = step[_STEP_NAME]
        status = step_statuses[parent] if parent in step_statuses else "Not Executed"
        G.node(parent, color=STEP_COLORS[status], style="filled")
        for child in step[_OUT_BOUND_EDGES]:
            child_name = child[_NEXT_STEP_NAME]
            if (parent, child_name) in display_edges:
                edge = child.get(_EDGE_LABEL, None)
                G.edge(parent, child_name, label=edge)

    return G


@attr.s
class Pipeline(Entity):
    """Pipeline for workflow.

    Attributes:
        name (str): The name of the pipeline.
        parameters (Sequence[Parameter]): The list of the parameters.
        pipeline_experiment_config (Optional[PipelineExperimentConfig]): If set,
            the workflow will attempt to create an experiment and trial before
            executing the steps. Creation will be skipped if an experiment or a trial with
            the same name already exists. By default, pipeline name is used as
            experiment name and execution id is used as the trial name.
            If set to None, no experiment or trial will be created automatically.
        steps (Sequence[Union[Step, StepCollection]]): The list of the non-conditional steps
            associated with the pipeline. Any steps that are within the
            `if_steps` or `else_steps` of a `ConditionStep` cannot be listed in the steps of a
            pipeline. Of particular note, the workflow service rejects any pipeline definitions that
            specify a step in the list of steps of a pipeline and that step in the `if_steps` or
            `else_steps` of any `ConditionStep`.
        sagemaker_session (sagemaker.session.Session): Session object that manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            pipeline creates one using the default AWS configuration chain.
    """

    name: str = attr.ib(factory=str)
    parameters: Sequence[Parameter] = attr.ib(factory=list)
    pipeline_experiment_config: Optional[PipelineExperimentConfig] = attr.ib(
        default=PipelineExperimentConfig(
            ExecutionVariables.PIPELINE_NAME, ExecutionVariables.PIPELINE_EXECUTION_ID
        )
    )
    steps: Sequence[Union[Step, StepCollection]] = attr.ib(factory=list)
    sagemaker_session: Session = attr.ib(factory=Session)

    _version: str = "2020-12-01"
    _metadata: Dict[str, Any] = dict()

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        return {
            "Version": self._version,
            "Metadata": self._metadata,
            "Parameters": list_to_request(self.parameters),
            "PipelineExperimentConfig": self.pipeline_experiment_config.to_request()
            if self.pipeline_experiment_config is not None
            else None,
            "Steps": list_to_request(self.steps),
        }

    def create(
        self,
        role_arn: str,
        description: str = None,
        tags: List[Dict[str, str]] = None,
        parallelism_config: ParallelismConfiguration = None,
    ) -> Dict[str, Any]:
        """Creates a Pipeline in the Pipelines service.

        Args:
            role_arn (str): The role arn that is assumed by the pipeline to create step artifacts.
            description (str): A description of the pipeline.
            tags (List[Dict[str, str]]): A list of {"Key": "string", "Value": "string"} dicts as
                tags.
            parallelism_config (Optional[ParallelismConfiguration]): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline.

        Returns:
            A response dict from the service.
        """
        tags = _append_project_tags(tags)
        kwargs = self._create_args(role_arn, description, parallelism_config)
        update_args(
            kwargs,
            Tags=tags,
        )
        return self.sagemaker_session.sagemaker_client.create_pipeline(**kwargs)

    def _create_args(
        self, role_arn: str, description: str, parallelism_config: ParallelismConfiguration
    ):
        """Constructs the keyword argument dict for a create_pipeline call.

        Args:
            role_arn (str): The role arn that is assumed by pipelines to create step artifacts.
            description (str): A description of the pipeline.
            parallelism_config (Optional[ParallelismConfiguration]): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline.

        Returns:
            A keyword argument dict for calling create_pipeline.
        """
        pipeline_definition = self.definition()
        kwargs = dict(
            PipelineName=self.name,
            RoleArn=role_arn,
        )

        # If pipeline definition is large, upload to S3 bucket and
        # provide PipelineDefinitionS3Location to request instead.
        if len(pipeline_definition.encode("utf-8")) < 1024 * 100:
            kwargs["PipelineDefinition"] = pipeline_definition
        else:
            desired_s3_uri = s3.s3_path_join(
                "s3://", self.sagemaker_session.default_bucket(), self.name
            )
            s3.S3Uploader.upload_string_as_file_body(
                body=pipeline_definition,
                desired_s3_uri=desired_s3_uri,
                sagemaker_session=self.sagemaker_session,
            )
            kwargs["PipelineDefinitionS3Location"] = {
                "Bucket": self.sagemaker_session.default_bucket(),
                "ObjectKey": self.name,
            }

        update_args(
            kwargs, PipelineDescription=description, ParallelismConfiguration=parallelism_config
        )
        return kwargs

    def describe(self) -> Dict[str, Any]:
        """Describes a Pipeline in the Workflow service.

        Returns:
            Response dict from the service. See `boto3 client documentation
            <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/\
sagemaker.html#SageMaker.Client.describe_pipeline>`_
        """
        return self.sagemaker_session.sagemaker_client.describe_pipeline(PipelineName=self.name)

    def display(self):
        """Prints out a Graphviz DAG visual for the Pipeline

        Returns:
            A Graphviz object representing the pipeline, if successful.
        """

        pipelineGraph = PipelineGraph.from_pipeline(self)
        adjacencyList = pipelineGraph.adjacency_list_with_edge_labels
        edges = generate_display_edges(adjacencyList)
        stepStatuses = {}

        return build_visual_dag(
            pipeline_name=self.name,
            adjacency_list=adjacencyList,
            step_statuses=stepStatuses,
            display_edges=edges,
        )

    def update(
        self,
        role_arn: str,
        description: str = None,
        parallelism_config: ParallelismConfiguration = None,
    ) -> Dict[str, Any]:
        """Updates a Pipeline in the Workflow service.

        Args:
            role_arn (str): The role arn that is assumed by pipelines to create step artifacts.
            description (str): A description of the pipeline.
            parallelism_config (Optional[ParallelismConfiguration]): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline.

        Returns:
            A response dict from the service.
        """
        kwargs = self._create_args(role_arn, description, parallelism_config)
        return self.sagemaker_session.sagemaker_client.update_pipeline(**kwargs)

    def upsert(
        self,
        role_arn: str,
        description: str = None,
        tags: List[Dict[str, str]] = None,
        parallelism_config: ParallelismConfiguration = None,
    ) -> Dict[str, Any]:
        """Creates a pipeline or updates it, if it already exists.

        Args:
            role_arn (str): The role arn that is assumed by workflow to create step artifacts.
            description (str): A description of the pipeline.
            tags (List[Dict[str, str]]): A list of {"Key": "string", "Value": "string"} dicts as
                tags.
            parallelism_config (Optional[Config for parallel steps, Parallelism configuration that
                is applied to each of. the executions

        Returns:
            response dict from service
        """
        exists = True
        try:
            self.describe()
        except ClientError as e:
            err = e.response.get("Error", {})
            if err.get("Code", None) == "ResourceNotFound":
                exists = False
            else:
                raise e

        if not exists:
            response = self.create(role_arn, description, tags, parallelism_config)
        else:
            response = self.update(role_arn, description)
            if tags is not None:
                old_tags = self.sagemaker_session.sagemaker_client.list_tags(
                    ResourceArn=response["PipelineArn"]
                )["Tags"]

                tag_keys = [tag["Key"] for tag in tags]
                for old_tag in old_tags:
                    if old_tag["Key"] not in tag_keys:
                        tags.append(old_tag)

                self.sagemaker_session.sagemaker_client.add_tags(
                    ResourceArn=response["PipelineArn"], Tags=tags
                )
        return response

    def get_last_execution(self, successful: bool = False, pipeline_arn: str = None):
        """Retrieve the last execution of a specified execution status

        Args:
            successful (bool): Desired status of the last execution retrieved
            pipeline_arn (str): Pipeline arn of a desired pipeline execution

        Returns:
            _PipelineExecution object
        """

        if pipeline_arn is None:
            pipeline = self.sagemaker_session.sagemaker_client.describe_pipeline(
                PipelineName=self.name
            )
            pipeline_arn = pipeline["PipelineArn"]

        search_expression = {"Filters": []}
        search_expression["Filters"].append(
            {"Name": "PipelineArn", "Operator": "Equals", "Value": pipeline_arn}
        )

        if successful:
            search_expression["Filters"].append(
                {"Name": "PipelineExecutionStatus", "Operator": "Equals", "Value": "Succeeded"}
            )

        search_args = {"Resource": "PipelineExecution", "SearchExpression": search_expression}

        search_response = self.sagemaker_session.sagemaker_client.search(**search_args)
        execution_arn = search_response["Results"][0]["PipelineExecution"]["PipelineExecutionArn"]
        return _PipelineExecution(arn=execution_arn, pipeline=self)

    def list_executions(self, next_token: str = None, max_results: int = 100):
        """Returns a list of executions done by the current pipeline

        Args:
            next_token (str): If the result of the previous call was truncated, a token that can be used to retrieve
                the next set of pipeline executions

        Returns:
            List of _PipelineExecution objects

        """
        if next_token is None:
            response = self.sagemaker_session.sagemaker_client.list_pipeline_executions(
                PipelineName=self.name, MaxResults=max_results
            )
        else:
            response = self.sagemaker_session.sagemaker_client.list_pipeline_executions(
                PipelineName=self.name, NextToken=next_token, MaxResults=max_results
            )
        pipelineExecutionList = []
        pipelineExecutionSummaries = response["PipelineExecutionSummaries"]
        nextToken = response.get("NextToken", None)

        for pipelineExecutionSummary in pipelineExecutionSummaries:
            pipelineExecutionList.append(
                _PipelineExecution(
                    arn=pipelineExecutionSummary["PipelineExecutionArn"],
                    sagemaker_session=self.sagemaker_session,
                    pipeline=self,
                )
            )

        return _ExecutionList(pipeline_executions=pipelineExecutionList, next_token=nextToken)

    def delete(self) -> Dict[str, Any]:
        """Deletes a Pipeline in the Workflow service.

        Returns:
            A response dict from the service.
        """
        return self.sagemaker_session.sagemaker_client.delete_pipeline(PipelineName=self.name)

    def start(
        self,
        parameters: Dict[str, Union[str, bool, int, float]] = None,
        execution_display_name: str = None,
        execution_description: str = None,
        parallelism_config: ParallelismConfiguration = None,
    ):
        """Starts a Pipeline execution in the Workflow service.

        Args:
            parameters (Dict[str, Union[str, bool, int, float]]): values to override
                pipeline parameters.
            execution_display_name (str): The display name of the pipeline execution.
            execution_description (str): A description of the execution.
            parallelism_config (Optional[ParallelismConfiguration]): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline.

        Returns:
            A `_PipelineExecution` instance, if successful.
        """
        kwargs = dict(PipelineName=self.name)
        update_args(
            kwargs,
            PipelineParameters=format_start_parameters(parameters),
            PipelineExecutionDescription=execution_description,
            PipelineExecutionDisplayName=execution_display_name,
            ParallelismConfiguration=parallelism_config,
        )
        response = self.sagemaker_session.sagemaker_client.start_pipeline_execution(**kwargs)
        return _PipelineExecution(
            arn=response["PipelineExecutionArn"],
            sagemaker_session=self.sagemaker_session,
            pipeline=self,
        )

    def definition(self) -> str:
        """Converts a request structure to string representation for workflow service calls."""
        request_dict = self.to_request()
        self._interpolate_step_collection_name_in_depends_on(request_dict["Steps"])
        request_dict["PipelineExperimentConfig"] = interpolate(
            request_dict["PipelineExperimentConfig"], {}, {}
        )
        callback_output_to_step_map = _map_callback_outputs(self.steps)
        lambda_output_to_step_name = _map_lambda_outputs(self.steps)
        request_dict["Steps"] = interpolate(
            request_dict["Steps"],
            callback_output_to_step_map=callback_output_to_step_map,
            lambda_output_to_step_map=lambda_output_to_step_name,
        )

        return json.dumps(request_dict)

    def _interpolate_step_collection_name_in_depends_on(self, step_requests: dict):
        """Insert step names as per `StepCollection` name in depends_on list

        Args:
            step_requests (dict): The raw step request dict without any interpolation.
        """
        step_name_map = {s.name: s for s in self.steps}
        for step_request in step_requests:
            if not step_request.get("DependsOn", None):
                continue
            depends_on = []
            for depend_step_name in step_request["DependsOn"]:
                if isinstance(step_name_map[depend_step_name], StepCollection):
                    depends_on.extend([s.name for s in step_name_map[depend_step_name].steps])
                else:
                    depends_on.append(depend_step_name)
            step_request["DependsOn"] = depends_on


def format_start_parameters(parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Formats start parameter overrides as a list of dicts.

    This list of dicts adheres to the request schema of:

        `{"Name": "MyParameterName", "Value": "MyValue"}`

    Args:
        parameters (Dict[str, Any]): A dict of named values where the keys are
            the names of the parameters to pass values into.
    """
    if parameters is None:
        return None
    return [{"Name": name, "Value": str(value)} for name, value in parameters.items()]


def interpolate(
    request_obj: RequestType,
    callback_output_to_step_map: Dict[str, str],
    lambda_output_to_step_map: Dict[str, str],
) -> RequestType:
    """Replaces Parameter values in a list of nested Dict[str, Any] with their workflow expression.

    Args:
        request_obj (RequestType): The request dict.
        callback_output_to_step_map (Dict[str, str]): A dict of output name -> step name.
        lambda_output_to_step_map (Dict[str, str]): A dict of output name -> step name.

    Returns:
        RequestType: The request dict with Parameter values replaced by their expression.
    """
    try:
        request_obj_copy = deepcopy(request_obj)
        return _interpolate(
            request_obj_copy,
            callback_output_to_step_map=callback_output_to_step_map,
            lambda_output_to_step_map=lambda_output_to_step_map,
        )
    except TypeError as type_err:
        raise TypeError("Not able to interpolate Pipeline definition: %s" % type_err)


def _interpolate(
    obj: Union[RequestType, Any],
    callback_output_to_step_map: Dict[str, str],
    lambda_output_to_step_map: Dict[str, str],
):
    """Walks the nested request dict, replacing Parameter type values with workflow expressions.

    Args:
        obj (Union[RequestType, Any]): The request dict.
        callback_output_to_step_map (Dict[str, str]): A dict of output name -> step name.
    """
    if isinstance(obj, (Expression, Parameter, Properties)):
        return obj.expr

    if isinstance(obj, CallbackOutput):
        step_name = callback_output_to_step_map[obj.output_name]
        return obj.expr(step_name)
    if isinstance(obj, LambdaOutput):
        step_name = lambda_output_to_step_map[obj.output_name]
        return obj.expr(step_name)
    if isinstance(obj, dict):
        new = obj.__class__()
        for key, value in obj.items():
            new[key] = interpolate(value, callback_output_to_step_map, lambda_output_to_step_map)
    elif isinstance(obj, (list, set, tuple)):
        new = obj.__class__(
            interpolate(value, callback_output_to_step_map, lambda_output_to_step_map)
            for value in obj
        )
    else:
        return obj
    return new


def _map_callback_outputs(steps: List[Step]):
    """Iterate over the provided steps, building a map of callback output parameters to step names.

    Args:
        step (List[Step]): The steps list.
    """

    callback_output_map = {}
    for step in steps:
        if isinstance(step, CallbackStep):
            if step.outputs:
                for output in step.outputs:
                    callback_output_map[output.output_name] = step.name

    return callback_output_map


def _map_lambda_outputs(steps: List[Step]):
    """Iterate over the provided steps, building a map of lambda output parameters to step names.

    Args:
        step (List[Step]): The steps list.
    """

    lambda_output_map = {}
    for step in steps:
        if isinstance(step, LambdaStep):
            if step.outputs:
                for output in step.outputs:
                    lambda_output_map[output.output_name] = step.name

    return lambda_output_map


def update_args(args: Dict[str, Any], **kwargs):
    """Updates the request arguments dict with a value, if populated.

    This handles the case when the service API doesn't like NoneTypes for argument values.

    Args:
        request_args (Dict[str, Any]): The request arguments dict.
        kwargs: key, value pairs to update the args dict with.
    """
    for key, value in kwargs.items():
        if value is not None:
            args.update({key: value})


class ImmutablePipeline(Pipeline):
    """ImmutablePipeline to support pipelines that should be immutable."""

    def display(self, pipeline_arn: str = None):
        """Prints out a Directed Acyclic Graph visual of the Pipeline

        Args:
            pipeline_arn (str): The pipeline arn for the desired pipeline to display.

        Returns:
            A Graphviz object representing the pipeline, if successful.
        """
        if pipeline_arn is None:
            pipeline = self.sagemaker_session.sagemaker_client.describe_pipeline(
                PipelineName=self.name
            )
            pipeline_arn = pipeline["PipelineArn"]
        response = self.sagemaker_session.sagemaker_client.describe_pipeline_graph(
            PipelineArn=pipeline_arn
        )
        adjacencyList = response["AdjacencyList"]
        edges = generate_display_edges(adjacencyList)
        stepStatuses = {}

        return build_visual_dag(
            pipeline_name=self.name,
            adjacency_list=adjacencyList,
            step_statuses=stepStatuses,
            display_edges=edges,
        )

    def update(
        self,
        role_arn: str,
        description: str = None,
        parallelism_config: ParallelismConfiguration = None,
    ):
        """Prevents an update on an ImmutablePipeline in the Workflow service.

        Args:
            role_arn (str): The role arn that is assumed by pipelines to create step artifacts.
            description (str): A description of the pipeline. (Defaults to None)
            parallelism_config (ParallelismConfiguration): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline. (Defaults to None)

        Returns:
            Exception
        """
        raise RuntimeError("Immutable Pipelines cannot be updated")

    def upsert(
        self,
        role_arn: str,
        description: str = None,
        tags: List[Dict[str, str]] = None,
        parallelism_config: ParallelismConfiguration = None,
    ):
        """Prevents an update on an ImmutablePipeline in the Workflow service.

        Args:
            role_arn (str): The role arn that is assumed by pipelines to create step artifacts.
            description (str): A description of the pipeline. (Defaults to None)
            tags (List[Dict[str, str]]): A list of {"Key": "string", "Value": "string"} dicts as
                tags. (Defaults to None)
            parallelism_config (ParallelismConfiguration): Parallelism configuration
                that is applied to each of the executions of the pipeline. It takes precedence
                over the parallelism configuration of the parent pipeline. (Defaults to None)

        Returns:
            Exception
        """
        raise RuntimeError("Immutable Pipelines cannot be updated")


@attr.s
class _PipelineList:
    """PipelineList class to encapsulate a list of Pipeline objects

    Attributes:
        pipelines (List[Pipeline]): A list of Pipeline objects
        next_token (str): If the result of the previous call was truncated, a token that can be used to retrieve
            the next set of Pipeline objects
    """

    pipelines: Sequence[Pipeline] = attr.ib(factory=list)
    next_token: str = attr.ib(default=None)


@attr.s
class _PipelineExecution:
    """Internal class for encapsulating pipeline execution instances.

    Attributes:
        arn (str): The arn of the pipeline execution.
        sagemaker_session (sagemaker.session.Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            pipeline creates one using the default AWS configuration chain.
        pipeline (Pipeline): Pipeline object tied to the current _PipelineExecution object
    """

    arn: str = attr.ib()
    sagemaker_session: Session = attr.ib(factory=Session)
    pipeline: Pipeline = attr.ib(default=None)

    def stop(self):
        """Stops a pipeline execution."""
        return self.sagemaker_session.sagemaker_client.stop_pipeline_execution(
            PipelineExecutionArn=self.arn
        )

    def describe(self):
        """Describes a pipeline execution.

        Returns:
             Information about the pipeline execution. See
             `boto3 client describe_pipeline_execution
             <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/\
sagemaker.html#SageMaker.Client.describe_pipeline_execution>`_.
        """
        return self.sagemaker_session.sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=self.arn,
        )

    def display(self):
        """Prints out a Graphviz DAG visual for a Pipeline's execution

        Returns
            A Graphviz object representing the execution, if Successful
        """

        if isinstance(self.pipeline, ImmutablePipeline):
            pipeline = self.sagemaker_session.sagemaker_client.describe_pipeline(
                PipelineName=self.pipeline.name
            )
            pipeline_arn = pipeline["PipelineArn"]
            response = self.sagemaker_session.sagemaker_client.describe_pipeline_graph(
                PipelineArn=pipeline_arn
            )
            adjacencyList = response["AdjacencyList"]
        else:
            pipelineGraph = PipelineGraph.from_pipeline(self.pipeline)
            adjacencyList = pipelineGraph.adjacency_list_with_edge_labels

        step_statuses = {}
        execution_steps = self.list_steps()
        edges = generate_display_edges(adjacencyList)
        for step in execution_steps:
            step_statuses[step[_STEP_NAME]] = step["StepStatus"]

        return build_visual_dag(
            pipeline_name=self.pipeline.name,
            adjacency_list=adjacencyList,
            step_statuses=step_statuses,
            display_edges=edges,
        )

    def list_steps(self):
        """Describes a pipeline execution's steps.

        Returns:
             Information about the steps of the pipeline execution. See
             `boto3 client list_pipeline_execution_steps
             <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/\
sagemaker.html#SageMaker.Client.list_pipeline_execution_steps>`_.
        """
        response = self.sagemaker_session.sagemaker_client.list_pipeline_execution_steps(
            PipelineExecutionArn=self.arn
        )
        return response["PipelineExecutionSteps"]

    def wait(self, delay=30, max_attempts=60):
        """Waits for a pipeline execution.

        Args:
            delay (int): The polling interval. (Defaults to 30 seconds)
            max_attempts (int): The maximum number of polling attempts.
                (Defaults to 60 polling attempts)
        """
        waiter_id = "PipelineExecutionComplete"
        # TODO: this waiter should be included in the botocore
        model = botocore.waiter.WaiterModel(
            {
                "version": 2,
                "waiters": {
                    waiter_id: {
                        "delay": delay,
                        "maxAttempts": max_attempts,
                        "operation": "DescribePipelineExecution",
                        "acceptors": [
                            {
                                "expected": "Succeeded",
                                "matcher": "path",
                                "state": "success",
                                "argument": "PipelineExecutionStatus",
                            },
                            {
                                "expected": "Failed",
                                "matcher": "path",
                                "state": "failure",
                                "argument": "PipelineExecutionStatus",
                            },
                        ],
                    }
                },
            }
        )
        waiter = botocore.waiter.create_waiter_with_client(
            waiter_id, model, self.sagemaker_session.sagemaker_client
        )
        waiter.wait(PipelineExecutionArn=self.arn)


@attr.s
class _ExecutionList:
    """ExecutionList class to encapsulate a list of _PipelineExecution objects

    Attributes:
        pipeline_executions (List[_PipelineExecution]): A list of _PipelineExecution objects
        next_token (str): If the result of the previous call was truncated, a token that can be used to retrieve
            the next set of _PipelineExecution objects
    """

    pipeline_executions: Sequence[_PipelineExecution] = attr.ib(factory=list)
    next_token: str = attr.ib(default=None)


class PipelineGraph:
    """Helper class representing the Pipeline Directed Acyclic Graph (DAG)

    Attributes:
        steps (Sequence[Union[Step, StepCollection]]): Sequence of `Step`s and/or `StepCollection`s
            that represent each node in the pipeline DAG
    """

    def __init__(self, steps: Sequence[Union[Step, StepCollection]]):
        self.step_map = {}
        self._generate_step_map(steps)
        self.adjacency_list = self._initialize_adjacency_list()
        self.adjacency_list_with_edge_labels = self.build_adjacency_list_with_condition_edges()
        if self.is_cyclic():
            raise ValueError("Cycle detected in pipeline step graph.")

    def _generate_step_map(self, steps: Sequence[Union[Step, StepCollection]]):
        """Helper method to create a mapping from Step/Step Collection name to itself."""
        for step in steps:
            if step.name in self.step_map:
                raise ValueError("Pipeline steps cannot have duplicate names. {}".format(step.name))
            self.step_map[step.name] = step
            if isinstance(step, ConditionStep):
                self._generate_step_map(step.if_steps + step.else_steps)
            if isinstance(step, StepCollection):
                self._generate_step_map(step.steps)

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline):
        """Create a PipelineGraph object from the Pipeline object."""
        return cls(pipeline.steps)

    def _initialize_adjacency_list(self) -> Dict[str, List[str]]:
        """Generate an adjacency list representing the step dependency DAG in this pipeline."""

        dependency_list = defaultdict(set)
        for step in self.step_map.values():
            if isinstance(step, Step):
                dependency_list[step.name].update(step._find_step_dependencies(self.step_map))

            if isinstance(step, ConditionStep):
                for child_step in step.if_steps + step.else_steps:
                    if isinstance(child_step, Step):
                        dependency_list[child_step.name].add(step.name)
                    elif isinstance(child_step, StepCollection):
                        child_first_step = self.step_map[child_step.name].steps[0].name
                        dependency_list[child_first_step].add(step.name)

        adjacency_list = {}
        for step in dependency_list:
            for step_dependency in dependency_list[step]:
                adjacency_list[step_dependency] = list(
                    set(adjacency_list.get(step_dependency, []) + [step])
                )
        for step in dependency_list:
            if step not in adjacency_list:
                adjacency_list[step] = []
        return adjacency_list

    def is_cyclic(self) -> bool:
        """Check if this pipeline graph is cyclic.

        Returns true if it is cyclic, false otherwise.
        """

        def is_cyclic_helper(current_step):
            visited_steps.add(current_step)
            recurse_steps.add(current_step)
            for child_step in self.adjacency_list[current_step]:
                if child_step in recurse_steps:
                    return True
                if child_step not in visited_steps:
                    if is_cyclic_helper(child_step):
                        return True
            recurse_steps.remove(current_step)
            return False

        visited_steps = set()
        recurse_steps = set()
        for step in self.adjacency_list:
            if step not in visited_steps:
                if is_cyclic_helper(step):
                    return True
        return False

    def build_adjacency_list_with_condition_edges(self) -> List[Dict[str, any]]:
        """Generates an adjacency list that includes edge labels for Condition Steps"""

        adjacency_list = []
        old_adjacency_list = self.adjacency_list

        if_edges = defaultdict(set)
        else_edges = defaultdict(set)

        for step in self.step_map.values():
            if isinstance(step, ConditionStep):
                for child_step in step.if_steps:
                    if isinstance(child_step, Step):
                        if_edges[step.name].add(child_step.name)
                    elif isinstance(child_step, StepCollection):
                        if_edges[step.name].add(self.step_map[child_step.name].steps[0].name)
                for child_step in step.else_steps:
                    if isinstance(child_step, Step):
                        else_edges[step.name].add(child_step.name)
                    elif isinstance(child_step, StepCollection):
                        else_edges[step.name].add(self.step_map[child_step.name].steps[0].name)

        for step in old_adjacency_list:
            adjacency_list_step = {}

            out_bound_edges = []
            for child_step in old_adjacency_list[step]:
                out_bound_edge = {_NEXT_STEP_NAME: child_step}
                if step in if_edges and child_step in if_edges[step]:
                    out_bound_edge[_EDGE_LABEL] = "True"
                elif step in else_edges and child_step in else_edges[step]:
                    out_bound_edge[_EDGE_LABEL] = "False"
                else:
                    out_bound_edge[_EDGE_LABEL] = None
                out_bound_edges.append(out_bound_edge)

            adjacency_list_step[_STEP_NAME] = step
            adjacency_list_step[_OUT_BOUND_EDGES] = out_bound_edges
            adjacency_list.append(adjacency_list_step)
        return adjacency_list

    def __iter__(self):
        """Perform topological sort traversal of the Pipeline Graph."""

        def topological_sort(current_step):
            visited_steps.add(current_step)
            for child_step in self.adjacency_list[current_step]:
                if child_step not in visited_steps:
                    topological_sort(child_step)
            self.stack.append(current_step)

        visited_steps = set()
        self.stack = []  # pylint: disable=W0201
        for step in self.adjacency_list:
            if step not in visited_steps:
                topological_sort(step)
        return self

    def __next__(self) -> Step:
        """Return the next Step node from the Topological sort order."""

        while self.stack:
            return self.step_map.get(self.stack.pop())
        raise StopIteration
