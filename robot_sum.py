
import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

class RoboSummarizer:

    def __init__(self, chkpt_path="/Users/byronwallace/code/RoboSum/weights/pl_title_/pl_title_2048.ckpt"):
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        self.config = BartConfig.from_pretrained('facebook/bart-large-cnn')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        # increase position embeddings from 1024 to 2048
        self.add_position_embeddings()

        # now add special tokens (for title and abstract demarcation)
        # as a general note: we'll assume "abstract" is either the 
        # actual abstract of extracted text from the same (i.e., punchlines)
        self.add_special_tokens()

        # now load the checkpoint
        print("loading checkpoint", chkpt_path)
        checkpoint = torch.load(chkpt_path,map_location="cpu")
        print("done")

        cnew={}
        for key, value in checkpoint['state_dict'].items():
          cnew[".".join(key.split('.')[1:])]=value
        self.model.load_state_dict(cnew)



    def add_position_embeddings(self, max_pos=2048):
        self.tokenizer.model_max_length=max_pos
        self.tokenizer.init_kwargs['model_max_length'] = max_pos
        current_max_pos, embed_size = self.model.model.encoder.embed_positions.weight.shape
        max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
        self.config.max_position_embeddings = max_pos
        assert (max_pos > current_max_pos)
        
        new_pos_embed = self.model.model.encoder.embed_positions.weight.new_empty(max_pos, embed_size)
        k = 2
        step = current_max_pos - 2
        while k < max_pos - 1:
            new_pos_embed[k:(k + step)] = self.model.model.encoder.embed_positions.weight[2:]
            k += step
        
        self.model.model.encoder.embed_positions.weight.data = new_pos_embed
        
        print("embedding position size increased to {}".format(self.config.max_position_embeddings))


    def add_special_tokens(self):
        print(self.model.state_dict()['final_logits_bias'].shape)
        special_tokens_dict = {'additional_special_tokens': ["<T>","<ABS>"]}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(len(self.tokenizer))
        
        print('We have added', num_added_toks, 'tokens')
        self.model.resize_token_embeddings(len(self.tokenizer))
        print(self.model.state_dict()['final_logits_bias'].shape)
        #assert(self.model.state_dict()["model.model.encoder.embed_tokens.weight"].shape[0]==len(self.tokenizer))
        print("Embedding size increased to {}".format(len(self.tokenizer)))

    def assemble(self, articles, max_len=2048):
        assembled_str = []
        for a in articles:
            assembled_str.append("<ABS> " + a["abs"] + " <TI> " + a["ti"])
        return " ".join(assembled_str)

    def run_through_model(self, input_ : str) -> str:
        inputs = self.tokenizer.batch_encode_plus([input_], max_length=2048, return_tensors='pt')
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        return " ".join([self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])


    def summarize(self, articles: list) -> str: 
        '''
        Assumes articles are provided as a list of dictionaries comprising
        titles and abstracts (or just punchlines) as ['ti'] and ['abs'] fields.
        Returns the generated summary.
        '''
        # assemble articles into a single string for input
        summarizer_input = self.assemble(articles)

        # pass through model
        summary = self.run_through_model(summarizer_input)

        return summary


'''
example = [{"ti": "Long-term follow-up in 257 ICA occlusion: comparison between EIAB-treated and untreated patients.", "abs": "In both groups, homogeneous by sex, age, neurological grading distribution and length of follow-up, the following parameters were considered: the incidence of ischaemic recurrences during the follow-up period; the characters of the recurrences with particular reference to the fatal stroke; the rate of ischaemic events per year."},
            {"ti": "Surgical and nonsurgical treatment of total carotid artery occlusion.", "abs": "Symptomatic occlusions occurred in 72.6 percent of the patients, the reconstructed group (46 patients) having a greater number of symptomatic vessels than the nonreconstructed group (63 patients) (p less than 0.05)."},
            {"ti": "Medical versus surgical treatment of patients with cerebrovascular insufficiency. A retrospective comparative study.", "abs": "The patients were divided in two groups, one being medically treated with anticoagulants, the other operated on either a carotid artery thrombosis by thromboendarterectomy or occlusion treated with an extracranial-intracranial arterial bypass." },
            {"ti": "Long-term assessment of cerebral perfusion following STA-MCA by-pass in patients.", "abs": "All patients had proximal occlusion of one internal carotid artery or intracranial occlusive disease of the internal carotid or middle cerebral arteries."},
            {"ti": "Comparison of the clinical results of STA-MCA anastomosis and the medical treatment in the cerebral low perfusion patients with viable brain tissue.", "abs":"The incidence of ipsilateral cerebral ischaemia was significantly low in the surgical group."},
            {"ti": "Evaluation of vasomotor reactivity by transcranial Doppler and acetazolamide test before and after extracranial-intracranial bypass in patients with internal carotid artery occlusion.", "abs":"aseline values of the mean blood flow velocity at rest on the affected side were reduced in both groups compared with the contralateral healthy side (group A, 46.0 +/- "},
            {"ti": "Superficial temporal artery--middle cerebral artery anastomosis for acute cerebral ischemia: the effect of small augmentation of blood flow.", "abs": "Subgroup analyses indicated that final outcomes for patients with mild to moderate paresis on admission were significantly better in the surgical group than in the non-surgical group (94% vs. 53%, p < 0.01)."},
            {"ti": "Measurements of regional cerebral blood flow in patients following superficial temporal artery-middle cerebral artery anastomosis.", "abs":"Compared to the non-surgical patients mean rCBF at this time was higher over both hemipheres. "}, 
            {"ti": "Clinical results of extracranial-intracranial bypass surgery in patients with hemodynamic cerebrovascular disease.", "abs": "Based on these results in 44 patients, the probability that successful surgery reduces the occurrence of ipsilateral ischemic stroke 1 year later was calculated."},
            {"ti": "STA-MCA bypass for symptomatic carotid occlusion and haemodynamic impairment.", "abs":"Cerebral perfusion assessed with SPECT scan improved in 88% of patients."},
            {"ti": "Treatment of the totally occluded carotid artery.", "abs":"The average time span from diagnosis of carotid occlusion until death was 4.75 years in the nonsurgical group and 4.52 years in the surgical group."},
            {"ti": "Overall management of vascular lesions considered treatable with extracranial-intracranial bypass: part 1. Internal carotid occlusion.", "abs":"By an average of 3 years after treatment began, 30 of 49 (61%) reached the same end points."}, 
            {"ti": "Long-term clinical and neurophysiological effects of reconstructive vascular surgery for cerebral ischemia.", "abs": "Clinical improvement occurred in all groups."},
            {"ti": "Evaluation of extracranial-to-intracranial bypass surgery using iodine 123 iodoamphetamine single-photon emission computed tomography.", "abs": "There was no increase in cerebral blood flow in one case with no operation."},
            {"ti": "Failure of extracranial-intracranial arterial bypass to reduce the risk of ischemic stroke. Results of an international randomized trial. The EC/IC Bypass Study Group.", "abs": "Two important subgroups of patients fared substantially worse in the surgical group: those with severe middle-cerebral-artery stenosis (n = 109, Mantel-Haenszel chi-square = 4.74), and those with persistence of ischemic symptoms after an internal-carotid-artery occlusion had been demonstrated (n = 287, chi-square = 4.04)."},
            {"ti": "STA-MCA bypass surgery for internal carotid artery occlusion--comparative follow-up study.","abs":"Long-term follow-up showed that there was no significant difference in the outcomes."},
            {"ti": "Perfusion MRI before and after acetazolamide administration for assessment of cerebrovascular reserve capacity in patients with symptomatic internal carotid artery (ICA) occlusion: comparison with 99mTc-ECD SPECT.", "abs":"Perfusion MRI before and after acetazolamide administration compares favourably with (99m)Tc-ECD SPECT for the detection of impaired CVR."},
         ]

rs = RoboSummarizer()
summary = rs.summarize(example)
print(summary)
'''
            
            
