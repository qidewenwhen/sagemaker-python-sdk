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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
from datetime import datetime
from unittest.mock import patch

from dateutil.tz import tzlocal
from mock import Mock
import pytest

from sagemaker import s3
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import (
    Pipeline,
    PipelineGraph,
    _PipelineExecution,
    ImmutablePipeline,
)
from sagemaker.workflow.parallelism_config import ParallelismConfiguration
from sagemaker.workflow.pipeline_experiment_config import (
    PipelineExperimentConfig,
    PipelineExperimentConfigProperties,
)
from sagemaker.workflow.step_collections import StepCollection
from tests.unit.sagemaker.workflow.helpers import (
    ordered,
    CustomStep,
    assert_adjacency_list_with_edges,
)


@pytest.fixture
def role_arn():
    return "arn:role"


@pytest.fixture
def sagemaker_session_mock():
    session_mock = Mock()
    session_mock.default_bucket = Mock(name="default_bucket", return_value="s3_bucket")
    return session_mock


def test_pipeline_create(sagemaker_session_mock, role_arn):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.create(role_arn=role_arn)
    assert sagemaker_session_mock.sagemaker_client.create_pipeline.called_with(
        PipelineName="MyPipeline", PipelineDefinition=pipeline.definition(), RoleArn=role_arn
    )


def test_pipeline_create_with_parallelism_config(sagemaker_session_mock, role_arn):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        pipeline_experiment_config=ParallelismConfiguration(max_parallel_execution_steps=10),
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.create(role_arn=role_arn)
    assert sagemaker_session_mock.sagemaker_client.create_pipeline.called_with(
        PipelineName="MyPipeline",
        PipelineDefinition=pipeline.definition(),
        RoleArn=role_arn,
        ParallelismConfiguration={"MaxParallelExecutionSteps": 10},
    )


def test_large_pipeline_create(sagemaker_session_mock, role_arn):
    parameter = ParameterString("MyStr")
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[CustomStep(name="MyStep", input_data=parameter)] * 2000,
        sagemaker_session=sagemaker_session_mock,
    )

    s3.S3Uploader.upload_string_as_file_body = Mock()

    pipeline.create(role_arn=role_arn)

    assert s3.S3Uploader.upload_string_as_file_body.called_with(
        body=pipeline.definition(), s3_uri="s3://s3_bucket/MyPipeline"
    )

    assert sagemaker_session_mock.sagemaker_client.create_pipeline.called_with(
        PipelineName="MyPipeline",
        PipelineDefinitionS3Location={"Bucket": "s3_bucket", "ObjectKey": "MyPipeline"},
        RoleArn=role_arn,
    )


def test_pipeline_update(sagemaker_session_mock, role_arn):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.update(role_arn=role_arn)
    assert sagemaker_session_mock.sagemaker_client.update_pipeline.called_with(
        PipelineName="MyPipeline", PipelineDefinition=pipeline.definition(), RoleArn=role_arn
    )


def test_pipeline_update_with_parallelism_config(sagemaker_session_mock, role_arn):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        pipeline_experiment_config=ParallelismConfiguration(max_parallel_execution_steps=10),
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.create(role_arn=role_arn)
    assert sagemaker_session_mock.sagemaker_client.update_pipeline.called_with(
        PipelineName="MyPipeline",
        PipelineDefinition=pipeline.definition(),
        RoleArn=role_arn,
        ParallelismConfiguration={"MaxParallelExecutionSteps": 10},
    )


def test_large_pipeline_update(sagemaker_session_mock, role_arn):
    parameter = ParameterString("MyStr")
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[CustomStep(name="MyStep", input_data=parameter)] * 2000,
        sagemaker_session=sagemaker_session_mock,
    )

    s3.S3Uploader.upload_string_as_file_body = Mock()

    pipeline.create(role_arn=role_arn)

    assert s3.S3Uploader.upload_string_as_file_body.called_with(
        body=pipeline.definition(), s3_uri="s3://s3_bucket/MyPipeline"
    )

    assert sagemaker_session_mock.sagemaker_client.update_pipeline.called_with(
        PipelineName="MyPipeline",
        PipelineDefinitionS3Location={"Bucket": "s3_bucket", "ObjectKey": "MyPipeline"},
        RoleArn=role_arn,
    )


def test_pipeline_upsert(sagemaker_session_mock, role_arn):
    sagemaker_session_mock.sagemaker_client.describe_pipeline.return_value = {
        "PipelineArn": "pipeline-arn"
    }
    sagemaker_session_mock.sagemaker_client.update_pipeline.return_value = {
        "PipelineArn": "pipeline-arn"
    }
    sagemaker_session_mock.sagemaker_client.list_tags.return_value = {
        "Tags": [{"Key": "dummy", "Value": "dummy_tag"}]
    }

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )

    tags = [
        {"Key": "foo", "Value": "abc"},
        {"Key": "bar", "Value": "xyz"},
    ]
    pipeline.upsert(role_arn=role_arn, tags=tags)

    sagemaker_session_mock.sagemaker_client.create_pipeline.assert_not_called()

    assert sagemaker_session_mock.sagemaker_client.update_pipeline.called_with(
        PipelineName="MyPipeline", PipelineDefinition=pipeline.definition(), RoleArn=role_arn
    )
    assert sagemaker_session_mock.sagemaker_client.list_tags.called_with(
        ResourceArn="mock_pipeline_arn"
    )

    tags.append({"Key": "dummy", "Value": "dummy_tag"})
    assert sagemaker_session_mock.sagemaker_client.add_tags.called_with(
        ResourceArn="mock_pipeline_arn", Tags=tags
    )


def test_pipeline_delete(sagemaker_session_mock):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.delete()
    assert sagemaker_session_mock.sagemaker_client.delete_pipeline.called_with(
        PipelineName="MyPipeline",
    )


def test_pipeline_describe(sagemaker_session_mock):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.describe()
    assert sagemaker_session_mock.sagemaker_client.describe_pipeline.called_with(
        PipelineName="MyPipeline",
    )


def test_immutable_pipeline_update(sagemaker_session_mock):
    step1 = CustomStep("Step1")
    pipeline = ImmutablePipeline(
        name="MyPipeline",
        parameters=[],
        steps=[step1],
        sagemaker_session=sagemaker_session_mock,
    )

    with pytest.raises(Exception) as error:
        pipeline.update(role_arn="")

    assert str(error.value) == "Immutable Pipelines cannot be updated"


def test_immutable_pipeline_upsert(sagemaker_session_mock):
    step1 = CustomStep("Step1")
    pipeline = ImmutablePipeline(
        name="MyPipeline",
        parameters=[],
        steps=[step1],
        sagemaker_session=sagemaker_session_mock,
    )

    with pytest.raises(Exception) as error:
        pipeline.upsert(role_arn="")

    assert str(error.value) == "Immutable Pipelines cannot be updated"


def test_pipeline_build_adjacency_list_with_condition_edges_without_condition_steps(
    sagemaker_session_mock,
):
    step1 = CustomStep(
        name="MyStep1",
        input_data=[
            [],  # parameter reference
            ExecutionVariables.PIPELINE_EXECUTION_ID,  # execution variable
            PipelineExperimentConfigProperties.EXPERIMENT_NAME,  # experiment config property
        ],
    )
    step2 = CustomStep(
        name="MyStep2", input_data=[step1.properties.ModelArtifacts.S3ModelArtifacts]
    )  # step property

    step3 = CustomStep(
        name="MyStep3", input_data=[step2.properties.ModelArtifacts.S3ModelArtifacts]
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[step1, step2, step3],
        sagemaker_session=sagemaker_session_mock,
    )

    pipelineGraph = PipelineGraph.from_pipeline(pipeline)
    output = pipelineGraph.adjacency_list_with_edge_labels

    expected = [
        {
            "StepName": "MyStep1",
            "OutBoundEdges": [{"NextStepName": "MyStep2", "EdgeLabel": None}],
        },
        {
            "StepName": "MyStep2",
            "OutBoundEdges": [{"NextStepName": "MyStep3", "EdgeLabel": None}],
        },
        {"StepName": "MyStep3", "OutBoundEdges": []},
    ]

    assert_adjacency_list_with_edges(output, expected)


def test_pipeline_build_adjacency_list_with_condition_edges_with_condition_step(
    sagemaker_session_mock,
):
    ifStep1 = CustomStep("IfStep1")
    ifStep2 = CustomStep("IfStep2")
    elseStep1 = CustomStep("ElseStep1")
    elseStep2 = CustomStep("ElseStep2")
    normalStep1 = CustomStep(
        "NormalStep", input_data=[elseStep2.properties.ModelArtifacts.S3ModelArtifacts]
    )

    conditionStep = ConditionStep(
        name="ConditionStep",
        conditions=[],
        if_steps=[ifStep1, ifStep2],
        else_steps=[elseStep1, elseStep2],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[conditionStep, normalStep1],
        sagemaker_session=sagemaker_session_mock,
    )

    pipelineGraph = PipelineGraph.from_pipeline(pipeline)
    output = pipelineGraph.adjacency_list_with_edge_labels

    expected = [
        {
            "StepName": "ConditionStep",
            "OutBoundEdges": [
                {"NextStepName": "ElseStep2", "EdgeLabel": "False"},
                {"NextStepName": "ElseStep1", "EdgeLabel": "False"},
                {"NextStepName": "IfStep2", "EdgeLabel": "True"},
                {"NextStepName": "IfStep1", "EdgeLabel": "True"},
            ],
        },
        {
            "StepName": "ElseStep2",
            "OutBoundEdges": [{"NextStepName": "NormalStep", "EdgeLabel": None}],
        },
        {"StepName": "IfStep1", "OutBoundEdges": []},
        {"StepName": "IfStep2", "OutBoundEdges": []},
        {"StepName": "ElseStep1", "OutBoundEdges": []},
        {"StepName": "NormalStep", "OutBoundEdges": []},
    ]

    assert_adjacency_list_with_edges(output, expected)


def test_pipeline_build_adjacency_list_with_condition_edges_with_step_collection(
    sagemaker_session_mock,
):
    step1 = CustomStep(
        name="MyStep1",
        input_data=[
            [],
            ExecutionVariables.PIPELINE_EXECUTION_ID,  # execution variable
            PipelineExperimentConfigProperties.EXPERIMENT_NAME,  # experiment config property
        ],
    )
    step2 = CustomStep(
        name="MyStep2", input_data=[step1.properties.ModelArtifacts.S3ModelArtifacts]
    )  # step property

    step_collection = StepCollection(name="MyStepCollection", steps=[step1, step2])

    conditionStep = ConditionStep(
        name="ConditionStep",
        conditions=[],
        if_steps=[step_collection],
        else_steps=[],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[conditionStep],
        sagemaker_session=sagemaker_session_mock,
    )

    pipelineGraph = PipelineGraph.from_pipeline(pipeline)
    output = pipelineGraph.adjacency_list_with_edge_labels

    expected = [
        {
            "StepName": "ConditionStep",
            "OutBoundEdges": [
                {"NextStepName": "MyStep1", "EdgeLabel": "True"},
            ],
        },
        {
            "StepName": "MyStep1",
            "OutBoundEdges": [{"NextStepName": "MyStep2", "EdgeLabel": None}],
        },
        {"StepName": "MyStep2", "OutBoundEdges": []},
    ]

    assert_adjacency_list_with_edges(output, expected)


@patch("sagemaker.workflow.pipeline.build_visual_dag")
def test_sdk_pipeline_display_with_redundant_edge(build_visual_dag, sagemaker_session_mock):
    ifStep1 = CustomStep("IfStep1")
    ifStep2 = CustomStep("IfStep2")
    elseStep1 = CustomStep("ElseStep1")
    elseStep2 = CustomStep("ElseStep2")
    normalStep1 = CustomStep(
        "NormalStep", input_data=[elseStep2.properties.ModelArtifacts.S3ModelArtifacts]
    )

    conditionStep = ConditionStep(
        name="ConditionStep",
        conditions=[],
        if_steps=[ifStep1, ifStep2, normalStep1],
        else_steps=[elseStep1, elseStep2],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[conditionStep],
        sagemaker_session=sagemaker_session_mock,
    )

    pipeline.display()

    actual_adj_list = PipelineGraph.from_pipeline(pipeline).adjacency_list_with_edge_labels

    expected = [
        {
            "StepName": "ConditionStep",
            "OutBoundEdges": [
                {"NextStepName": "ElseStep2", "EdgeLabel": "False"},
                {"NextStepName": "ElseStep1", "EdgeLabel": "False"},
                {"NextStepName": "IfStep2", "EdgeLabel": "True"},
                {"NextStepName": "IfStep1", "EdgeLabel": "True"},
                {"NextStepName": "NormalStep", "EdgeLabel": "True"},
            ],
        },
        {
            "StepName": "ElseStep2",
            "OutBoundEdges": [{"NextStepName": "NormalStep", "EdgeLabel": None}],
        },
        {"StepName": "IfStep1", "OutBoundEdges": []},
        {"StepName": "IfStep2", "OutBoundEdges": []},
        {"StepName": "ElseStep1", "OutBoundEdges": []},
        {"StepName": "NormalStep", "OutBoundEdges": []},
    ]

    edges = set(
        [
            ("ConditionStep", "ElseStep2"),
            ("ConditionStep", "ElseStep1"),
            ("ConditionStep", "IfStep2"),
            ("ConditionStep", "IfStep1"),
            ("ElseStep2", "NormalStep"),
        ]
    )

    assert_adjacency_list_with_edges(actual_adj_list, expected)
    build_visual_dag.assert_called_with(
        pipeline_name="MyPipeline",
        adjacency_list=actual_adj_list,
        step_statuses={},
        display_edges=edges,
    )


def test_get_last_execution(sagemaker_session_mock):
    step1 = CustomStep(
        name="MyStep1",
        input_data=[
            [],
            ExecutionVariables.PIPELINE_EXECUTION_ID,  # execution variable
            PipelineExperimentConfigProperties.EXPERIMENT_NAME,  # experiment config property
        ],
    )
    step2 = CustomStep(
        name="MyStep2", input_data=[step1.properties.ModelArtifacts.S3ModelArtifacts]
    )  # step property

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[step1, step2],
        sagemaker_session=sagemaker_session_mock,
    )

    sagemaker_session_mock.sagemaker_client.describe_pipeline.return_value = {
        "PipelineArn": "pipeline-arn"
    }

    sagemaker_session_mock.sagemaker_client.search.return_value = {
        "Results": [
            {
                "PipelineExecution": {
                    "PipelineArn": "pipeline-arn",
                    "PipelineExecutionArn": "pipeline-execution",
                    "PipelineExecutionStatus": "Succeeded",
                    "LastModifiedTime": datetime(2022, 7, 12, 17, 52, 19, 433000, tzinfo=tzlocal()),
                }
            },
            {
                "PipelineExecution": {
                    "PipelineArn": "pipeline-arn2",
                    "PipelineExecutionArn": "pipeline-execution2",
                    "PipelineExecutionStatus": "Failed",
                    "LastModifiedTime": datetime(2022, 7, 12, 17, 52, 20, 433000, tzinfo=tzlocal()),
                }
            },
        ]
    }
    output = pipeline.get_last_execution()

    expected = _PipelineExecution(arn="pipeline-execution", pipeline=pipeline)

    assert output.arn == expected.arn
    assert output.pipeline == expected.pipeline
    assert isinstance(output, _PipelineExecution)


@patch("sagemaker.workflow.pipeline.build_visual_dag")
@patch.object(_PipelineExecution, "list_steps")
def test_pipeline_execution_display(list_steps, build_visual_dag, sagemaker_session_mock):
    step1 = CustomStep(
        name="MyStep1",
        input_data=[
            [],
            ExecutionVariables.PIPELINE_EXECUTION_ID,  # execution variable
            PipelineExperimentConfigProperties.EXPERIMENT_NAME,  # experiment config property
        ],
    )
    step2 = CustomStep(
        name="MyStep2", input_data=[step1.properties.ModelArtifacts.S3ModelArtifacts]
    )  # step property

    step3 = CustomStep(
        name="MyStep3", input_data=[step2.properties.ModelArtifacts.S3ModelArtifacts]
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[step1, step2, step3],
        sagemaker_session=sagemaker_session_mock,
    )

    execution = _PipelineExecution(
        arn="arn", sagemaker_session=pipeline.sagemaker_session, pipeline=pipeline
    )

    list_steps.return_value = [
        {
            "StepName": "MyStep1",
            "StartTime": datetime(2022, 7, 12, 17, 52, 19, 433000, tzinfo=tzlocal()),
            "StepStatus": "Succeeded",
            "AttemptCount": 0,
            "Metadata": {},
        },
        {
            "StepName": "MyStep2",
            "StartTime": datetime(2022, 7, 12, 17, 52, 20, 433000, tzinfo=tzlocal()),
            "StepStatus": "Failed",
            "AttemptCount": 0,
            "Metadata": {},
        },
    ]

    step_statuses = {"MyStep1": "Succeeded", "MyStep2": "Failed"}

    execution.display()

    actual_adj_list = PipelineGraph.from_pipeline(pipeline).adjacency_list_with_edge_labels

    expected = [
        {
            "StepName": "MyStep1",
            "OutBoundEdges": [{"NextStepName": "MyStep2", "EdgeLabel": None}],
        },
        {
            "StepName": "MyStep2",
            "OutBoundEdges": [{"NextStepName": "MyStep3", "EdgeLabel": None}],
        },
        {"StepName": "MyStep3", "OutBoundEdges": []},
    ]

    edges = set([("MyStep1", "MyStep2"), ("MyStep2", "MyStep3")])

    assert_adjacency_list_with_edges(actual_adj_list, expected)
    build_visual_dag.assert_called_with(
        pipeline_name="MyPipeline",
        adjacency_list=actual_adj_list,
        step_statuses=step_statuses,
        display_edges=edges,
    )


@patch("sagemaker.workflow.pipeline.build_visual_dag")
def test_immutable_pipeline_display(build_visual_dag, sagemaker_session_mock):
    pipeline = ImmutablePipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )

    describeGraphResponse = {
        "PipelineName": "MyPipeline",
        "AdjacencyList": [
            {
                "StepName": "MyStep1",
                "OutBoundEdges": [{"NextStepName": "MyStep2"}],
            },
            {
                "StepName": "MyStep2",
                "OutBoundEdges": [{"NextStepName": "MyStep3"}],
            },
            {"StepName": "MyStep3", "OutBoundEdges": []},
        ],
    }

    sagemaker_session_mock.sagemaker_client.describe_pipeline.return_value = {
        "PipelineArn": "pipeline-arn"
    }
    sagemaker_session_mock.sagemaker_client.describe_pipeline_graph.return_value = (
        describeGraphResponse
    )

    edges = set([("MyStep1", "MyStep2"), ("MyStep2", "MyStep3")])

    pipeline.display()

    build_visual_dag.assert_called_with(
        pipeline_name="MyPipeline",
        adjacency_list=describeGraphResponse["AdjacencyList"],
        step_statuses={},
        display_edges=edges,
    )


@patch("sagemaker.workflow.pipeline.build_visual_dag")
@patch.object(_PipelineExecution, "list_steps")
def test_immutable_pipeline_execution_display(list_steps, build_visual_dag, sagemaker_session_mock):
    pipeline = ImmutablePipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )

    execution = _PipelineExecution(
        arn="arn", sagemaker_session=pipeline.sagemaker_session, pipeline=pipeline
    )

    describeGraphResponse = {
        "PipelineName": "MyPipeline",
        "AdjacencyList": [
            {
                "StepName": "MyStep1",
                "OutBoundEdges": [{"NextStepName": "MyStep2"}],
            },
            {
                "StepName": "MyStep2",
                "OutBoundEdges": [{"NextStepName": "MyStep3"}],
            },
            {"StepName": "MyStep3", "OutBoundEdges": []},
        ],
    }

    sagemaker_session_mock.sagemaker_client.describe_pipeline.return_value = {
        "PipelineArn": "pipeline-arn"
    }
    sagemaker_session_mock.sagemaker_client.describe_pipeline_graph.return_value = (
        describeGraphResponse
    )

    list_steps.return_value = [
        {
            "StepName": "MyStep1",
            "StartTime": datetime(2022, 7, 12, 17, 52, 19, 433000, tzinfo=tzlocal()),
            "StepStatus": "Succeeded",
            "AttemptCount": 0,
            "Metadata": {},
        },
        {
            "StepName": "MyStep2",
            "StartTime": datetime(2022, 7, 12, 17, 52, 20, 433000, tzinfo=tzlocal()),
            "StepStatus": "Failed",
            "AttemptCount": 0,
            "Metadata": {},
        },
    ]

    execution.display()

    edges = set([("MyStep1", "MyStep2"), ("MyStep2", "MyStep3")])
    step_statuses = {"MyStep1": "Succeeded", "MyStep2": "Failed"}

    build_visual_dag.assert_called_with(
        pipeline_name="MyPipeline",
        adjacency_list=describeGraphResponse["AdjacencyList"],
        step_statuses=step_statuses,
        display_edges=edges,
    )


def test_pipeline_start(sagemaker_session_mock):
    sagemaker_session_mock.sagemaker_client.start_pipeline_execution.return_value = {
        "PipelineExecutionArn": "my:arn"
    }
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[ParameterString("alpha", "beta"), ParameterString("gamma", "delta")],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.start()
    assert sagemaker_session_mock.start_pipeline_execution.called_with(
        PipelineName="MyPipeline",
    )

    pipeline.start(execution_display_name="pipeline-execution")
    assert sagemaker_session_mock.start_pipeline_execution.called_with(
        PipelineName="MyPipeline", PipelineExecutionDisplayName="pipeline-execution"
    )

    pipeline.start(parameters=dict(alpha="epsilon"))
    assert sagemaker_session_mock.start_pipeline_execution.called_with(
        PipelineName="MyPipeline", PipelineParameters=[{"Name": "alpha", "Value": "epsilon"}]
    )


def test_pipeline_basic():
    parameter = ParameterString("MyStr")
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[CustomStep(name="MyStep", input_data=parameter)],
        sagemaker_session=sagemaker_session_mock,
    )
    assert pipeline.to_request() == {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [{"Name": "MyStr", "Type": "String"}],
        "PipelineExperimentConfig": {
            "ExperimentName": ExecutionVariables.PIPELINE_NAME,
            "TrialName": ExecutionVariables.PIPELINE_EXECUTION_ID,
        },
        "Steps": [{"Name": "MyStep", "Type": "Training", "Arguments": {"input_data": parameter}}],
    }
    assert ordered(json.loads(pipeline.definition())) == ordered(
        {
            "Version": "2020-12-01",
            "Metadata": {},
            "Parameters": [{"Name": "MyStr", "Type": "String"}],
            "PipelineExperimentConfig": {
                "ExperimentName": {"Get": "Execution.PipelineName"},
                "TrialName": {"Get": "Execution.PipelineExecutionId"},
            },
            "Steps": [
                {
                    "Name": "MyStep",
                    "Type": "Training",
                    "Arguments": {"input_data": {"Get": "Parameters.MyStr"}},
                }
            ],
        }
    )


def test_pipeline_two_step(sagemaker_session_mock):
    parameter = ParameterString("MyStr")
    step1 = CustomStep(
        name="MyStep1",
        input_data=[
            parameter,  # parameter reference
            ExecutionVariables.PIPELINE_EXECUTION_ID,  # execution variable
            PipelineExperimentConfigProperties.EXPERIMENT_NAME,  # experiment config property
        ],
    )
    step2 = CustomStep(
        name="MyStep2", input_data=[step1.properties.ModelArtifacts.S3ModelArtifacts]
    )  # step property
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[step1, step2],
        sagemaker_session=sagemaker_session_mock,
    )
    assert pipeline.to_request() == {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [{"Name": "MyStr", "Type": "String"}],
        "PipelineExperimentConfig": {
            "ExperimentName": ExecutionVariables.PIPELINE_NAME,
            "TrialName": ExecutionVariables.PIPELINE_EXECUTION_ID,
        },
        "Steps": [
            {
                "Name": "MyStep1",
                "Type": "Training",
                "Arguments": {
                    "input_data": [
                        parameter,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        PipelineExperimentConfigProperties.EXPERIMENT_NAME,
                    ]
                },
            },
            {
                "Name": "MyStep2",
                "Type": "Training",
                "Arguments": {"input_data": [step1.properties.ModelArtifacts.S3ModelArtifacts]},
            },
        ],
    }
    assert ordered(json.loads(pipeline.definition())) == ordered(
        {
            "Version": "2020-12-01",
            "Metadata": {},
            "Parameters": [{"Name": "MyStr", "Type": "String"}],
            "PipelineExperimentConfig": {
                "ExperimentName": {"Get": "Execution.PipelineName"},
                "TrialName": {"Get": "Execution.PipelineExecutionId"},
            },
            "Steps": [
                {
                    "Name": "MyStep1",
                    "Type": "Training",
                    "Arguments": {
                        "input_data": [
                            {"Get": "Parameters.MyStr"},
                            {"Get": "Execution.PipelineExecutionId"},
                            {"Get": "PipelineExperimentConfig.ExperimentName"},
                        ]
                    },
                },
                {
                    "Name": "MyStep2",
                    "Type": "Training",
                    "Arguments": {
                        "input_data": [{"Get": "Steps.MyStep1.ModelArtifacts.S3ModelArtifacts"}]
                    },
                },
            ],
        }
    )

    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered({"MyStep1": ["MyStep2"], "MyStep2": []})


def test_pipeline_override_experiment_config():
    pipeline = Pipeline(
        name="MyPipeline",
        pipeline_experiment_config=PipelineExperimentConfig("MyExperiment", "MyTrial"),
        steps=[CustomStep(name="MyStep", input_data="input")],
        sagemaker_session=sagemaker_session_mock,
    )
    assert ordered(json.loads(pipeline.definition())) == ordered(
        {
            "Version": "2020-12-01",
            "Metadata": {},
            "Parameters": [],
            "PipelineExperimentConfig": {"ExperimentName": "MyExperiment", "TrialName": "MyTrial"},
            "Steps": [
                {
                    "Name": "MyStep",
                    "Type": "Training",
                    "Arguments": {"input_data": "input"},
                }
            ],
        }
    )


def test_pipeline_disable_experiment_config():
    pipeline = Pipeline(
        name="MyPipeline",
        pipeline_experiment_config=None,
        steps=[CustomStep(name="MyStep", input_data="input")],
        sagemaker_session=sagemaker_session_mock,
    )
    assert ordered(json.loads(pipeline.definition())) == ordered(
        {
            "Version": "2020-12-01",
            "Metadata": {},
            "Parameters": [],
            "PipelineExperimentConfig": None,
            "Steps": [
                {
                    "Name": "MyStep",
                    "Type": "Training",
                    "Arguments": {"input_data": "input"},
                }
            ],
        }
    )


def test_pipeline_execution_basics(sagemaker_session_mock):
    sagemaker_session_mock.sagemaker_client.start_pipeline_execution.return_value = {
        "PipelineExecutionArn": "my:arn"
    }
    sagemaker_session_mock.sagemaker_client.list_pipeline_execution_steps.return_value = {
        "PipelineExecutionSteps": [Mock()]
    }
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[ParameterString("alpha", "beta"), ParameterString("gamma", "delta")],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    execution = pipeline.start()
    execution.stop()
    assert sagemaker_session_mock.sagemaker_client.stop_pipeline_execution.called_with(
        PipelineExecutionArn="my:arn"
    )
    execution.describe()
    assert sagemaker_session_mock.sagemaker_client.describe_pipeline_execution.called_with(
        PipelineExecutionArn="my:arn"
    )
    steps = execution.list_steps()
    assert sagemaker_session_mock.sagemaker_client.describe_pipeline_execution_steps.called_with(
        PipelineExecutionArn="my:arn"
    )
    assert len(steps) == 1
