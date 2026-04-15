import pathlib

import pytest

from codefinetuner.pipeline import _parse_args, _setup_logger, main, run_pipeline


# --- _parse_args ---

def test_parse_args_all_flags(mocker):
    test_args = [
        "pipeline.py",
        "--config", "test.yaml",
        "--log-level", "DEBUG",
        "--skip-preprocess",
        "--skip-finetune",
        "--skip-evaluate",
        "--skip-convert",
    ]
    mocker.patch("sys.argv", test_args)
    args = _parse_args()
    assert args.config == pathlib.Path("test.yaml")
    assert args.log_level == "DEBUG"
    assert args.skip_preprocess is True
    assert args.skip_finetune is True
    assert args.skip_evaluate is True
    assert args.skip_convert is True


def test_parse_args_missing_required_config(mocker):
    test_args = ["pipeline.py"]
    mocker.patch("sys.argv", test_args)
    with pytest.raises(SystemExit):
        _parse_args()


# --- run_pipeline ---

def test_run_pipeline_all_stages(tmp_path, mocker):
    preprocess_config_mock = mocker.patch("codefinetuner.pipeline.PreprocessConfig")
    preprocess_config_mock.load_from_yaml.return_value = "test_preprocess_config"
    preprocess_run_mock = mocker.patch("codefinetuner.pipeline.preprocess_run")

    finetune_config_mock = mocker.patch("codefinetuner.pipeline.FinetuneConfig")
    finetune_config_mock.load_from_yaml.return_value = "test_finetune_config"
    finetune_run_mock = mocker.patch("codefinetuner.pipeline.finetune_run")

    evaluate_config_mock = mocker.patch("codefinetuner.pipeline.EvaluateConfig")
    evaluate_config_mock.load_from_yaml.return_value = "test_evaluate_config"
    evaluate_run_mock = mocker.patch("codefinetuner.pipeline.evaluate_run")

    convert_config_mock = mocker.patch("codefinetuner.pipeline.ConvertConfig")
    convert_config_mock.load_from_yaml.return_value = "test_convert_config"
    convert_run_mock = mocker.patch("codefinetuner.pipeline.convert_run")

    tmp_cofig_path = tmp_path / "test_config.yaml"
    tmp_cofig_path.touch()

    run_pipeline(tmp_cofig_path)

    preprocess_config_mock.load_from_yaml.assert_called_once_with(tmp_cofig_path)
    preprocess_run_mock.assert_called_once_with("test_preprocess_config")
    
    finetune_config_mock.load_from_yaml.assert_called_once_with(tmp_cofig_path)
    finetune_run_mock.assert_called_once_with("test_finetune_config")
    
    evaluate_config_mock.load_from_yaml.assert_called_once_with(tmp_cofig_path)
    evaluate_run_mock.assert_called_once_with("test_evaluate_config")
    
    convert_config_mock.load_from_yaml.assert_called_once_with(tmp_cofig_path)
    convert_run_mock.assert_called_once_with("test_convert_config")


def test_run_pipeline_skip_all(tmp_path, mocker):
    mocker.patch("codefinetuner.pipeline.PreprocessConfig")
    preprocess_run_mock = mocker.patch("codefinetuner.pipeline.preprocess_run")
    mocker.patch("codefinetuner.pipeline.FinetuneConfig")
    finetune_run_mock = mocker.patch("codefinetuner.pipeline.finetune_run")
    mocker.patch("codefinetuner.pipeline.EvaluateConfig")
    evaluate_run_mock = mocker.patch("codefinetuner.pipeline.evaluate_run")
    mocker.patch("codefinetuner.pipeline.ConvertConfig")
    convert_run_mock = mocker.patch("codefinetuner.pipeline.convert_run")

    tmp_cofig_path = tmp_path / "test_config.yaml"
    tmp_cofig_path.touch()

    run_pipeline(
        tmp_cofig_path,
        skip_preprocess=True,
        skip_finetune=True,
        skip_evaluate=True,
        skip_convert=True
    )

    preprocess_run_mock.assert_not_called()
    finetune_run_mock.assert_not_called()
    evaluate_run_mock.assert_not_called()
    convert_run_mock.assert_not_called()


# --- main ---

def test_main_success(tmp_path, mocker):
    parse_args_mock = mocker.patch("codefinetuner.pipeline._parse_args")
    setup_logger_mock = mocker.patch("codefinetuner.pipeline._setup_logger")
    run_pipeline_mock = mocker.patch("codefinetuner.pipeline.run_pipeline")

    tmp_cofig_path = tmp_path / "test_config.yaml"
    tmp_cofig_path.touch()

    args_mock = mocker.MagicMock()
    args_mock.config = tmp_cofig_path
    args_mock.log_level = "INFO"
    args_mock.skip_preprocess = False
    args_mock.skip_finetune = False
    args_mock.skip_evaluate = False
    args_mock.skip_convert = False
    parse_args_mock.return_value = args_mock

    main()

    parse_args_mock.assert_called_once()
    setup_logger_mock.assert_called_once_with("INFO")
    run_pipeline_mock.assert_called_once_with(
        config_path=tmp_cofig_path,
        skip_preprocess=False,
        skip_finetune=False,
        skip_evaluate=False,
        skip_convert=False,
    )


def test_main_handles_exception(mocker):
    mocker.patch("codefinetuner.pipeline._parse_args")
    mocker.patch("codefinetuner.pipeline._setup_logger")
    run_pipeline_mock = mocker.patch("codefinetuner.pipeline.run_pipeline")
    logger_mock = mocker.patch("codefinetuner.pipeline.logger")
    exit_mock = mocker.patch("sys.exit")

    run_pipeline_mock.side_effect = Exception("Test Error")
    
    main()
    
    logger_mock.exception.assert_called_once_with("Pipeline execution failed")
    exit_mock.assert_called_once_with(1)
