terraform {

  backend "s3" {
    bucket = "customerintelligence"
    key    = "ml_platform/terraform.tfstate"
    region = "eu-central-1"
    endpoints = {
      s3 = "http://s3-de-central.profitbricks.com"
    }
    skip_credentials_validation = true
    skip_requesting_account_id  = true
  }

  required_providers {
    ionoscloud = {
      source  = "ionos-cloud/ionoscloud"
      version = ">= 6.4.10"
    }
  }
}

provider "ionoscloud" {
  # For his authorization to work, environment variable TF_VAR_ionos_token must be available 
  token = "${var.ionos_token}"
}


module "kubeflow_cluster_1" {
  source = "./kubeflow-cluster-module"

  datacenter_id = "${var.datacenter_id}"
  cluster_name  = "k8sDeveloperCluster"
  k8s_version   = "1.31.3"
  node_count    = 3
}