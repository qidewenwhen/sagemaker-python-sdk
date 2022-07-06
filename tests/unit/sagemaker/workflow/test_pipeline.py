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

import graphviz

from mock import Mock

from sagemaker import s3
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.parallelism_config import ParallelismConfiguration
from sagemaker.workflow.pipeline_experiment_config import (
    PipelineExperimentConfig,
    PipelineExperimentConfigProperties,
)
from tests.unit.sagemaker.workflow.helpers import ordered, CustomStep


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


def test_pipeline_display(sagemaker_session_mock):
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
    output = pipeline.display()

    expected = graphviz.Digraph("MyPipeline", strict=True)
    expected.node("MyStep1")
    expected.node("MyStep2")
    expected.node("MyStep3")
    expected.edge("MyStep1", "MyStep2", label=None)
    expected.edge("MyStep2", "MyStep3", label=None)

    assert sorted(output.body) == sorted(expected.body)


def test_pipeline_display_with_condition_steps(sagemaker_session_mock):
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

    output = pipeline.display()

    expected = graphviz.Digraph("MyPipeline")
    expected.node("IfStep1")
    expected.node("IfStep2")
    expected.node("ElseStep1")
    expected.node("ElseStep2")
    expected.node("NormalStep")
    expected.node("ConditionStep")
    expected.edge("ConditionStep", "IfStep1", label="True")
    expected.edge("ConditionStep", "IfStep2", label="True")
    expected.edge("ConditionStep", "ElseStep1", label="False")
    expected.edge("ConditionStep", "ElseStep2", label="False")
    expected.edge("ElseStep2", "NormalStep", label=None)

    assert sorted(output.body) == sorted(expected.body)


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
